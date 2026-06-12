# Vendored SDKs

Drop the **TensorRT-RTX SDK** Linux tarball here for the `nv-trt-rtx` build.

It's an auth-gated NVIDIA download (<https://developer.nvidia.com/tensorrt-rtx>), so it
can't be fetched inside Docker. `docker/Dockerfile.nv-trt-rtx` COPYs whatever
`TensorRT-RTX-*.tar.*` it finds in this directory, extracts it to `/opt/tensorrt-rtx`,
and builds the provider against it (`--tensorrt_rtx_home`).

- **Version:** `1.3.0.35` (CUDA 12.9), matching the Windows `tensorrt_rtx_1_3` runtime.
  Keep it in sync with `TENSORRT_RTX_VERSION` in `../VERSIONS.env`.
- **Expected file:** e.g. `TensorRT-RTX-1.3.0.35.Linux.x86_64-gnu.cuda-12.9.tar.gz`
- **Get it:** NVIDIA Developer → *TensorRT for RTX* → Linux x86_64, CUDA 12.9 tar.

The tarball is git-ignored (see `../.gitignore`). The lightweight `libtensorrt_rtx`
runtime is **not** harvested into `out/` — the app bundles it separately, exactly like
the Windows DLL.
