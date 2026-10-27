# SPDX-License-Identifier: Apache-2.0
"""Native Kunlun device management with CUDA-compatible computation.

XPUML queries do not initialize a CUDA context. Capability and execution use
the vendor's torch.cuda implementation through NonNvmlCudaPlatform.
"""

import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import torch_xmlir._XMLIRC as xpuml

from fastvideo.logger import init_logger
from fastvideo.platforms.cuda import NonNvmlCudaPlatform, device_id_to_physical_device_id

logger = init_logger(__name__)


def _check_result(result: Any, operation: str) -> None:
    # Initialization bindings return an enum in some torch_xmlir versions.
    if result is not None and int(result) != 0:
        raise RuntimeError(f"{operation} failed with error code {int(result)}")


@contextmanager
def _xpuml_context() -> Iterator[None]:
    _check_result(xpuml.xpumlInit(), "xpumlInit")
    try:
        yield
    finally:
        try:
            _check_result(xpuml.xpumlShutdown(), "xpumlShutdown")
        except Exception:
            logger.debug("XPUML shutdown failed", exc_info=True)


@lru_cache(maxsize=1)
def _xpuml_library() -> ctypes.CDLL:
    library = ctypes.CDLL("libxpuml.so.1")
    library.xpumlDeviceGetHandleByIndex.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    library.xpumlDeviceGetHandleByIndex.restype = ctypes.c_int
    return library


def _get_device_handle(physical_device_id: int) -> int:
    # Older _XMLIRC bindings copy/truncate opaque handles. Obtain the
    # library-owned pointer directly, without patching the installed runtime.
    handle = ctypes.c_void_p()
    _check_result(
        _xpuml_library().xpumlDeviceGetHandleByIndex(physical_device_id, ctypes.byref(handle)),
        "xpumlDeviceGetHandleByIndex",
    )
    if handle.value is None:
        raise RuntimeError("XPUML returned a null device handle")
    return handle.value


class XpumlCudaPlatform(NonNvmlCudaPlatform):
    """Keep CUDA dispatch and collectives, using XPUML for device metadata."""

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # The vendor process group implements backend="nccl" on XPU. PyNccl
        # calls NVIDIA's C library directly and cannot use that integration.
        return "fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"

    @classmethod
    def get_device_count(cls) -> int:
        with _xpuml_context():
            return int(xpuml.xpumlDeviceGetCount())

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        with _xpuml_context():
            name = xpuml.xpumlDeviceGetName(_get_device_handle(physical_device_id))
            if not name:
                raise RuntimeError("XPUML returned an empty device name")
            return name.decode() if isinstance(name, bytes) else str(name)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = device_id_to_physical_device_id(device_id)
        with _xpuml_context():
            info = xpuml.xpumlDeviceGetMemoryInfo(_get_device_handle(physical_device_id))
            # Native bindings name global memory separately from on-chip L3.
            total = int(info.totalGlobalMemory if hasattr(info, "totalGlobalMemory") else info.total)
            if total <= 0:
                raise RuntimeError("XPUML returned invalid total device memory")
            return total

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        with _xpuml_context():
            # The affected pybind xpumlDeviceGetUuid returns the handle itself.
            # The native API returns the persistent UUID string instead.
            library = _xpuml_library()
            library.xpumlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
            library.xpumlDeviceGetUUID.restype = ctypes.c_int
            uuid = ctypes.create_string_buffer(96)
            _check_result(
                library.xpumlDeviceGetUUID(_get_device_handle(physical_device_id), uuid, len(uuid)),
                "xpumlDeviceGetUUID",
            )
            if not uuid.value:
                raise RuntimeError("XPUML returned an empty device UUID")
            return uuid.value.decode()

    @classmethod
    def is_full_nvlink(cls, physical_device_ids: list[int]) -> bool:
        # XPU interconnects are not NVIDIA NVLink. Collective support is
        # provided independently by the CUDA-compatible process group.
        return False
