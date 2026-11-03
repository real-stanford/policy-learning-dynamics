"""Translate CUDA device indices to EGL device indices for MUJOCO_EGL_DEVICE_ID.

MUJOCO_EGL_DEVICE_ID indexes the eglQueryDevicesEXT() list (mujoco and
robosuite both do this, and mujoco's wrapper only asks for the first 10
devices). That list is in driver enumeration order, which need not match
CUDA / nvidia-smi ordering, and on multi-vendor hosts it also contains Mesa
and software devices. Passing a CUDA index straight through therefore sends
rendering to a different physical GPU than the rank computes on.

Resolution, in order of preference:

1. EGL_NV_device_cuda: NVIDIA EGL devices report their CUDA ordinal via
   eglQueryDeviceAttribEXT(EGL_CUDA_DEVICE_NV). This is the authoritative
   mapping and works even where the DRM queries below return nothing. The
   ordinal is numbered the way the CUDA runtime in *this process* would see
   it, i.e. relative to CUDA_VISIBLE_DEVICES (hidden GPUs report no value),
   so the physical index is first translated to its visible ordinal.
2. DRM node: CUDA index --(nvidia-smi)--> PCI bus id --(/dev/dri/by-path)-->
   card node --(EGL_DRM_DEVICE_FILE_EXT)--> EGL index. Only NVIDIA-vendor
   entries are considered so a Mesa entry for the same card is never chosen.
3. Identity, with a warning.

Only stdlib + ctypes so it can be imported from the isolated LIBERO stack.
"""

from __future__ import annotations

import ctypes
import functools
import os
import subprocess
import warnings
from dataclasses import dataclass

_EGL_EXTENSIONS = 0x3055
_EGL_DRM_DEVICE_FILE_EXT = 0x3233
_EGL_CUDA_DEVICE_NV = 0x323A
# mujoco.egl.egl_ext.eglQueryDevicesEXT(max_devices=10) is what robosuite
# indexes into; anything beyond this cannot be addressed via the env var.
_MUJOCO_MAX_DEVICES = 10


@dataclass(frozen=True)
class _EglDevice:
    index: int
    cuda_index: int | None
    drm_node: str | None
    is_nvidia: bool


@functools.lru_cache(maxsize=None)
def _enumerate_egl_devices() -> tuple[_EglDevice, ...]:
    libegl = ctypes.CDLL("libEGL.so.1")
    get_proc = libegl.eglGetProcAddress
    get_proc.restype = ctypes.c_void_p
    get_proc.argtypes = [ctypes.c_char_p]

    def proc(name: bytes, restype, *argtypes):
        addr = get_proc(name)
        if not addr:
            raise RuntimeError(f"{name.decode()} not available")
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

    query_devices = proc(
        b"eglQueryDevicesEXT",
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )
    query_string = proc(
        b"eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int
    )
    query_attrib = proc(
        b"eglQueryDeviceAttribEXT",
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_long),  # EGLAttrib is intptr_t
    )

    count = ctypes.c_int()
    devices = (ctypes.c_void_p * 64)()
    if not query_devices(64, devices, ctypes.byref(count)):
        raise RuntimeError("eglQueryDevicesEXT failed")

    out = []
    for i in range(count.value):
        ext = (query_string(devices[i], _EGL_EXTENSIONS) or b"").decode()
        is_nvidia = "EGL_NV_device_cuda" in ext
        cuda_index = None
        if is_nvidia:
            value = ctypes.c_long(-1)
            if query_attrib(devices[i], _EGL_CUDA_DEVICE_NV, ctypes.byref(value)):
                cuda_index = int(value.value)
        drm = query_string(devices[i], _EGL_DRM_DEVICE_FILE_EXT)
        out.append(
            _EglDevice(
                index=i,
                cuda_index=cuda_index,
                drm_node=drm.decode() if drm else None,
                is_nvidia=is_nvidia,
            )
        )
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _drm_node_by_cuda_index() -> dict[int, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    node_by_index: dict[int, str] = {}
    for line in out.strip().splitlines():
        index_str, bus_id = (part.strip() for part in line.split(","))
        domain, bus, rest = bus_id.lower().split(":")
        by_path = f"/dev/dri/by-path/pci-{domain[-4:]}:{bus}:{rest}-card"
        node_by_index[int(index_str)] = os.path.realpath(by_path)
    return node_by_index


def _visible_ordinal(cuda_index: int) -> int | None:
    """Position of physical `cuda_index` in CUDA_VISIBLE_DEVICES (identity if unset)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible == "":
        return cuda_index
    try:
        return [int(v) for v in visible.split(",")].index(cuda_index)
    except ValueError:  # non-integer entries (UUIDs) or index not visible
        return None


def _resolve(cuda_index: int) -> int:
    devices = _enumerate_egl_devices()

    # 1. Direct CUDA ordinal reported by the NVIDIA EGL driver (visible numbering).
    ordinal = _visible_ordinal(cuda_index)
    if ordinal is not None:
        for dev in devices:
            if dev.cuda_index == ordinal:
                return dev.index

    # 2. DRM card node, restricted to NVIDIA-vendor entries.
    drm_node = _drm_node_by_cuda_index()[cuda_index]
    for dev in devices:
        if dev.is_nvidia and dev.drm_node == drm_node:
            return dev.index

    raise LookupError(
        f"no NVIDIA EGL device matches CUDA device {cuda_index} "
        f"(enumerated {len(devices)} EGL devices)"
    )


def cuda_to_egl_device_index(cuda_index: int) -> int:
    """Map a physical CUDA device index to the matching EGL device index.

    Falls back to the identity mapping (with a warning) if resolution fails,
    which preserves the historical behavior of assuming the orders match.
    """
    try:
        egl_index = _resolve(cuda_index)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not resolve EGL device index for CUDA device {cuda_index} "
            f"({type(exc).__name__}: {exc}); assuming EGL order matches CUDA order.",
            stacklevel=2,
        )
        return cuda_index
    if egl_index >= _MUJOCO_MAX_DEVICES:
        warnings.warn(
            f"EGL device index {egl_index} for CUDA device {cuda_index} is beyond "
            f"the {_MUJOCO_MAX_DEVICES} devices mujoco enumerates; assuming EGL "
            "order matches CUDA order.",
            stacklevel=2,
        )
        return cuda_index
    return egl_index
