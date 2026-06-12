# Integrating the source-built EP libs into the app

The kit produces, per group, a `libonnxruntime.so` and its
`libonnxruntime_providers_*.so` siblings under `out/<group>/lib/`. This doc
covers how the app links/loads them and ships them in the bundle.

## How ORT finds a provider at runtime

These EPs are **not** statically linked. ORT's provider bridge
(`libonnxruntime_providers_shared.so`) `dlopen`s `libonnxruntime_providers_<ep>.so`
**by filename** the moment the app appends that EP to the session. So the only
hard requirement is: the provider `.so` for the selected EP sits on the
library search path next to `libonnxruntime.so`, together with
`libonnxruntime_providers_shared.so` and the vendor runtime libs it depends on.

That means one `libonnxruntime.so` can serve multiple provider `.so`s **of the
same ORT version**. You do not need a separate base runtime per vendor — you
need the right provider `.so` present.

## Two ways to point the `ort` crate at these libs

The relevant knobs in this fork's `ort-sys/build.rs` (verified):
`ORT_LIB_LOCATION`, `ORT_LIB_PROFILE`, `ORT_PREFER_DYNAMIC_LINK`,
`ORT_SKIP_DOWNLOAD`; and the `load-dynamic` runtime var `ORT_DYLIB_PATH`
(`src/lib.rs:82`).

### Option A — `load-dynamic` (recommended)

Enable the `load-dynamic` feature on `ort` in the app. Then nothing is linked
at build time; at startup the app sets `ORT_DYLIB_PATH` to the bundled
`libonnxruntime.so` and ORT loads everything (incl. provider libs) from beside
it. This fully decouples the app build from these artifacts — no ABI link
step, no per-vendor rebuild — and matches how the AppImage already ships libs
in `usr/lib`. Cleanest fit for a multi-EP bundle.

- App `Cargo.toml`: add `"load-dynamic"` to the `ort` features.
- App startup (before creating any ORT env/session): set `ORT_DYLIB_PATH` to
  the path of the bundled `libonnxruntime.so` (next to the executable / in
  `usr/lib`).
- Bundle: drop `out/<group>/lib/*` into the lib dir (see below).

### Option B — link against the dylib at build time

Build the app with:

```bash
export ORT_LIB_LOCATION=/abs/path/to/build-kit/out/<group>   # dir containing lib/
export ORT_PREFER_DYNAMIC_LINK=1                              # skip static-component linking
export ORT_SKIP_DOWNLOAD=1                                    # never fetch a prebuilt
```

`build.rs` then copies the dylibs from `<ORT_LIB_LOCATION>/lib/` next to the
output binary. The catch: one build = one `ORT_LIB_LOCATION` = one group, so
this is awkward for a multi-vendor bundle. Prefer Option A.

## Shipping in the AppImage

The app's `bundle_appimage.sh` already copies vendor runtime libs into
`usr/lib` (it has globs for `libonnxruntime_providers_rocm*.so` /
`_openvino*.so`, plus the ROCm/OpenVINO/CUDA runtime libs). Feed it from here:

1. Point the bundler's ORT lib source at `out/<group>/lib/` (or copy those
   `.so`s into the dir it harvests from) so the **base `libonnxruntime.so` and
   the provider `.so`s** land in `usr/lib`.
2. Keep the existing vendor-runtime bundling (ROCm `.so.6/.4/.1`, OpenVINO
   `.so.2024`, CUDA/cuDNN/TRT) — those are the libs the provider `.so` depends
   on. **Their major versions must match the base image this kit built
   against** (see README → "Version coupling"). Cross-check `MANIFEST.txt`.
3. For ROCm also ship the MIGraphX runtime (`libmigraphx*.so`) since the kit
   builds the MIGraphX provider alongside ROCm.

Per-vendor AppImage = that vendor's provider `.so` + base `libonnxruntime.so` +
`libonnxruntime_providers_shared.so` + the matching vendor runtime libs.

## NV-TensorRT-RTX (Linux x86_64)

Built by default in the `nv-trt-rtx` group, bringing Linux NVIDIA up to the
Windows RTX path (regular `cuda-trt` is also built, as the proven fallback).
The Rust side already exists in this fork (`NvTensorRTRTXExecutionProvider`,
`OrtNvTensorRtRtxProviderOptions`); the kit produces the matching
`libonnxruntime_providers_nv_tensorrt_rtx.so`.

**Runtime (ORT 1.23.2):** this provider links the lightweight **TensorRT-RTX SDK**
(`libtensorrt_rtx.so.1`), **not** full TensorRT — the Linux analogue of the Windows
`tensorrt_rtx_1_3.dll` path. The SDK is vendored into `build-kit/vendor/` and built via
`--tensorrt_rtx_home` (see `Dockerfile.nv-trt-rtx`). Confirm with
`ldd out/nv-trt-rtx/lib/libonnxruntime_providers_nv_tensorrt_rtx.so` — it should show
`libtensorrt_rtx`, not `libnvinfer`.

To use the NV EP for the app's Linux NVIDIA bundle: ship `out/nv-trt-rtx/lib/*`
(its `libonnxruntime.so` + the `_nv_tensorrt_rtx`/`_shared` providers) **plus the
TensorRT-RTX runtime + CUDA / cuDNN libs** (`libtensorrt_rtx.so.1`, `libcudart.so.12`,
`libcudnn*.so.9`), and select `NvTensorRTRTXExecutionProvider` in the EP factory's Linux
NVIDIA branch instead of the regular TensorRT EP. The `libtensorrt_rtx` runtime is **not**
harvested into `out/` — bundle it from the vendored SDK, exactly like the Windows DLL.

## Optional fork fix: fail loud instead of silent CPU fallback

The footgun that caused the original AMD crash lives at
`ort-sys/build.rs:539` — when a requested feature set has no matching
`dist.txt` row it prints a notice and downloads the CPU-only `none` build, so
the breakage only surfaces as a runtime crash later. Since we own this fork
(branch `linux-source-eps`), consider making that path **panic** with a clear
message ("EP X requested but no prebuilt for <target>; set ORT_LIB_LOCATION or
build from source — see build-kit/") whenever `ORT_LIB_LOCATION`/
`ORT_SKIP_DOWNLOAD` are not set. That converts a silent CPU downgrade into an
actionable build-time error. (Left undone pending your call — it changes
behavior for anyone relying on the current fallback.)
