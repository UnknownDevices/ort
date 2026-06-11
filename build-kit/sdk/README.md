# Offline staging for the Linux TensorRT-RTX SDK (optional)

**You normally don't need to touch this directory.** The `nv-trt-rtx` build
downloads the SDK automatically from the verified `TRT_RTX_URL` in
`VERSIONS.env` (unauthenticated NVIDIA developer download).

Use this directory **only for air-gapped / offline build boxes**: drop the
Linux x86_64 tarball here and it takes precedence over the download.

```
./sdk/TensorRT-RTX-1.3.0.35-linux-x86_64-cuda-12.9-Release-external.tar.gz
```

Keep the CUDA minor (12.9) consistent with `NV_TENSORRT_IMAGE`. Tarballs are
gitignored; do not commit them.
