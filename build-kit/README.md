# ONNX Runtime EP source-build kit

Compiles, from source, every ONNX Runtime execution-provider shared library
the pyke prebuilts don't ship, so the app can do AMD / Intel / NVIDIA(-RTX) /
arm64 inference instead of silently falling back to CPU and crashing.

One pass per box builds **everything that box's targets can use**, with full
optimization. Runs entirely in Docker so the hosts stay clean.

> Pinned to ONNX Runtime **v1.23.2** (matches the Windows side). The fork's prebuilts are
> `ms@1.22.0` (`ort-sys/dist.txt`), but ORT's C API is backward-compatible, so the `rc.10`
> bindings link fine against the 1.23.2 runtime built here. 1.23.2 is the floor for the
> `nv-trt-rtx` provider to link the lightweight TensorRT-RTX SDK (`libtensorrt_rtx`)
> instead of full TensorRT.
>
> **Exception — `rocm` stays on `v1.22.0`** (`ROCM_ORT_VERSION`): ONNX Runtime removed the
> ROCm EP in 1.23 (MIGraphX-only), so AMD keeps the ROCm EP at its last version. Bundles are
> per-vendor, so the mixed ORT versions never meet.

## What gets built

| Group | EPs in the build | x86_64 box | arm64 DGX | Notes |
|---|---|:--:|:--:|---|
| `cpu` | XNNPACK + oneDNN (x86) / +KleidiAI (arm) | ✅ | ✅ | dependency-light no-GPU fallback; oneDNN is x86-only |
| `rocm` | ROCm + MIGraphX + XNNPACK | ✅ | — | AMD; **pinned to ORT 1.22** (ROCm EP deleted in 1.23); arm box has no ROCm SDK |
| `openvino` | OpenVINO + XNNPACK | ✅ | — | Intel GPU/NPU/CPU |
| `cuda-trt` | CUDA + TensorRT + XNNPACK | ✅ | ✅ | arm64 fills the missing aarch64 cu12 |
| `nv-trt-rtx` | CUDA + NV-TensorRT-RTX + XNNPACK | ✅ | — | consumer RTX (x86_64 only); links the TensorRT-RTX SDK — vendor the SDK tarball into `vendor/` (see below) |
| `webgpu` | WebGPU (Dawn, over Vulkan) + XNNPACK | ✅ | — | cross-vendor GPU; heavy Dawn build; zero-copy interop unproven |

`x86_64` → `./build-amd64.sh` · `arm64` → `./build-arm64.sh`. Each build emits
`libonnxruntime.so` + its `libonnxruntime_providers_*.so` into
`out/<group>/lib/`, with a `MANIFEST.txt` (EP flags, sonames, sha256).

Out of scope (different machines/toolchains): macOS+iOS CoreML (a Mac),
Windows DirectML+TRT-RTX (already shipping off prebuilts), Android QNN (NDK).

## Optimizations applied

- **`--enable_lto`** on the CPU/ROCm/OpenVINO/WebGPU groups (`ORT_ENABLE_LTO=1`).
  **Off for the NVIDIA groups** (`CUDA_ENABLE_LTO=0`): CUDA device-LTO defers all
  per-arch `ptxas` codegen into a single, non-parallel `nvlink`, which serializes
  the whole 8-arch build onto one core (the multi-hour grind) and is the OOM
  source. With it off, the codegen fans out across every core instead.
- **Full CUDA arch coverage**, native SASS per generation + PTX for forward
  compat. x86_64: Turing→Blackwell (`75;80;86;89;90;100;120` + PTX); arm64:
  Xavier/Orin/Grace-Hopper (`72;87;90` + PTX). Hence the CUDA 12.8+ base image
  (Blackwell sm_120 = RTX 50 needs CUDA ≥ 12.8). Breadth is cheap in wall-clock
  now that LTO no longer serializes it — the only cost is a bigger shipped fatbin.
- **Bounded CUDA parallelism** — the NVIDIA groups run `CUDA_NPROC=64
  CUDA_NVCC_THREADS=2` instead of all-cores, because the 8-arch `ptxas` passes
  spike memory (~`NPROC*NVCC_THREADS` concurrent `ptxas` at ~2-5 GB each). Tuned
  for ~200 GB RAM; raise `CUDA_NPROC` after watching peak RSS.
- **XNNPACK in every group** (KleidiAI auto-links on aarch64) for a fast CPU path.
- **Release** config; non-CUDA groups parallel to all cores.
- **No `-march=native`** — deliberate. ORT's MLAS kernels are runtime-dispatched
  (AVX2/AVX512/NEON), so a generic build is already optimal *and* portable to
  end users' CPUs; pinning `-march` would bind to the build box.

## Prerequisites

- Docker with internet (pulls vendor base images; build fetches ORT source + deps).
- **Disk:** ~30–50 GB per group (many CUDA arches make the CUDA groups the
  heaviest). **Time:** the broad-arch CUDA build is large but now fans out across
  all cores (device-LTO off); `ccache` is on by default (`CCACHE_DIR_HOST=cache/
  ccache`, 50 GB cap) so a re-run after a failure skips the bulk of it.
- **RAM:** the NVIDIA groups are memory-bound, not core-bound — defaults assume
  ~200 GB. On less, lower `CUDA_NPROC` / `CUDA_NVCC_THREADS`.
- NVIDIA base pulls anonymously from `nvcr.io` (verified — no NGC login needed).
- **No manual downloads.** Every dependency is fetched by the build: base images
  (anonymous) and the ONNX Runtime source + its cmake deps. The NVIDIA groups link
  the TensorRT 10.x already inside the `nvcr` base image — no extra SDK to stage.
- A GPU is **not** required to *build* any of this.

## Run it

```bash
# x86_64 big-CPU box  (downloads what it needs: base images, ORT source + deps)
cd build-kit && ./build-amd64.sh

# arm64 NVIDIA DGX
cd build-kit && ./build-arm64.sh

./check_outputs.sh   # verify every expected provider .so landed
```

Common overrides (or edit `VERSIONS.env`):

```bash
NPROC=96 CCACHE_DIR_HOST=./cache/ccache ./build-amd64.sh
ORT_CUDA_ARCHS="90-real;90-virtual" ./build-arm64.sh   # narrow to just GH200
CUDA_NPROC=96 CUDA_NVCC_THREADS=2 ./build-amd64.sh      # more RAM -> push NVIDIA groups harder
CUDA_ENABLE_LTO=1 ./build-amd64.sh                      # re-enable CUDA LTO (slow, single-core link)
```

## Two NVIDIA version knobs to confirm on the box

These are the only values I can't pin without the hardware/portal — align them
once in `VERSIONS.env` and everything else follows:

1. **`NV_TENSORRT_IMAGE`** — full-TensorRT base for the `cuda-trt` group only. Default
   `25.06-py3` = CUDA 12.9.1 + TRT 10.11. `nv-trt-rtx` does *not* use this — it bases on
   **`NV_RTX_CUDA_IMAGE`** (CUDA-only, default `nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04`)
   and links the vendored TensorRT-RTX SDK (drop the tarball in `vendor/`; see
   `vendor/README.md` and `TENSORRT_RTX_VERSION`). Check CUDA:
   `docker run --rm <img> cat /usr/local/cuda/version.json`.
2. **`ROCM_IMAGE` / `OPENVINO_IMAGE` versions** — the provider `.so` links these
   ABIs, so the app must bundle the **same-major** vendor runtime libs:

| Group | Built against | App bundles (same major) |
|---|---|---|
| `rocm` | ROCm 6.x | `libamdhip64.so.6`, `librocblas.so.4`, `libMIOpen.so.1`, `libmigraphx*.so` |
| `openvino` | OpenVINO 2025.x | `libopenvino.so.2510`, plugin `.so`s |
| `cuda-trt` | CUDA 12.9 / full TRT 10.11 | `libcudart.so.12`, `libcudnn*.so.9`, `libnvinfer.so.10` |
| `nv-trt-rtx` | CUDA 12.9 / TensorRT-RTX 1.3 | `libcudart.so.12`, `libcudnn*.so.9`, `libtensorrt_rtx.so.1` (lightweight RTX runtime, like the Windows `tensorrt_rtx_1_3.dll`) |

Cross-check sonames in each `MANIFEST.txt` against what `bundle_appimage.sh` ships.

## Troubleshooting

- **CMake too old** — handled (`pip install cmake==3.31.6`; ORT 1.23 needs ≥ 3.28).
- **`nv-trt-rtx` configure: "tensorrt_rtx_home … must be specified" / SDK not found** —
  the build needs the TensorRT-RTX SDK vendored into `vendor/` (auth-gated NVIDIA
  download); the Dockerfile COPYs + extracts it to `/opt/tensorrt-rtx` and passes
  `--tensorrt_rtx_home`. The base is CUDA-only on purpose: with full TensorRT also present,
  its `NvInfer.h` hijacks cmake's RTX-SDK header probe and the version parse fails.
- **ROCm EP is version-sensitive** — if it trips, match `ROCM_IMAGE` to a ROCm
  the ORT release is happy with, or add `--rocm_version 6.2` via `ORT_EXTRA_FLAGS`.
- **OpenVINO device flag** — built with `--use_openvino AUTO`; app picks the real
  device at runtime. If `AUTO` is rejected, edit `docker/Dockerfile.openvino`.
- **OpenVINO build: `fatal error: format: No such file or directory`** — ORT 1.23's
  OpenVINO EP compiles at C++20 and uses `<format>`, which libstdc++ ships only from GCC
  13+. The ubuntu22 OpenVINO image has GCC 11, so `Dockerfile.openvino` adds GCC 13 (the
  `ubuntu-toolchain-r/test` PPA) and sets `CC`/`CXX`. Consequence: the Intel AppImage must
  also bundle GCC 13's `libstdc++.so.6` (the EP needs `GLIBCXX_3.4.32` from `<format>`).
- **LTO trips a group** — set `ORT_ENABLE_LTO=0` to drop it for that run.

See `INTEGRATION.md` for how the app links/loads and ships these libs.
