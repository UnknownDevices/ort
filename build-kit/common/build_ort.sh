#!/usr/bin/env bash
# Builds ONNX Runtime for one EP group inside a container and harvests the
# shared libs into /out/<group>/lib. Driven by ORT_* env (set per Dockerfile).
set -euo pipefail

: "${ORT_VERSION:?}"
: "${ORT_GROUP:?}"
: "${ORT_EP_FLAGS:?}"
NPROC="${NPROC:-$(nproc)}"
SRC_DIR="${SRC_DIR:-/src/onnxruntime}"
BUILD_DIR="${BUILD_DIR:-/tmp/ort-build/${ORT_GROUP}}"
OUT_DIR="/out/${ORT_GROUP}"

echo "ONNX Runtime ${ORT_VERSION} | group=${ORT_GROUP} | arch=$(uname -m) | jobs=${NPROC}"
echo "EP flags: ${ORT_EP_FLAGS} ${ORT_EXTRA_FLAGS:-}"
echo "LTO=${ORT_ENABLE_LTO:-1} CUDA=${ORT_CUDA_ARCHS:-n/a} GFX=${ORT_ROCM_GFX:-n/a}"

if [ ! -d "${SRC_DIR}/.git" ]; then
   git clone --depth 1 --branch "${ORT_VERSION}" --recursive \
      https://github.com/microsoft/onnxruntime "${SRC_DIR}"
fi

# GitLab regenerates archive tarballs (compression changes), so the SHA1s ORT
# pins in cmake/deps.txt for GitLab-hosted deps (e.g. Eigen) drift and
# FetchContent aborts. Re-derive them from the served archive — the URL pins a
# commit, so only packaging changed, not the source.
deps="${SRC_DIR}/cmake/deps.txt"
if [ -f "$deps" ]; then
   grep 'gitlab.com' "$deps" | while IFS=';' read -r dname durl dhash _; do
      [ -n "$durl" ] || continue
      curl -fsSL --retry 3 -o /tmp/dep.bin "$durl" || continue
      dnew="$(sha1sum /tmp/dep.bin | cut -d' ' -f1)"; rm -f /tmp/dep.bin
      [ "$dnew" = "$dhash" ] && continue
      echo "deps.txt: ${dname} ${dhash} -> ${dnew}"
      awk -F';' -v n="$dname" -v h="$dnew" 'BEGIN{OFS=";"} $1==n{$3=h} 1' \
         "$deps" > "$deps.tmp" && mv "$deps.tmp" "$deps"
   done
fi

if [ -n "${CCACHE_DIR:-}" ] && command -v ccache >/dev/null 2>&1; then
   # Cache nvcc too, not just host C/C++ -- the CUDA objects are the bulk of a
   # re-run's cost. Bump max_size well past the 5G default so the multi-arch CUDA
   # objects don't evict each other across groups / re-runs.
   export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache \
          CMAKE_CUDA_COMPILER_LAUNCHER=ccache
   export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-50G}"
fi

# MIGraphX needs its cmake config (migraphx-dev). Since ORT 1.23 removed the
# ROCm EP, MIGraphX is the ONLY AMD GPU EP — dropping it would silently yield a
# CPU-only "AMD" provider, so a missing config is fatal, not a fallback.
# -L: /opt/rocm is a versioned symlink (-> /opt/rocm-7.x) on the ROCm image, and find's
# default -P won't descend into a symlinked start path, so it must follow links here.
if [[ "${ORT_EP_FLAGS}" == *use_migraphx* ]] && ! find -L /opt/rocm -name 'migraphx*onfig.cmake' 2>/dev/null | grep -q .; then
   echo "ERROR: MIGraphX cmake config not found under /opt/rocm (install migraphx-dev)." >&2
   echo "       ORT 1.23 has no ROCm EP fallback — refusing to build a CPU-only AMD provider." >&2
   exit 1
fi

EXTRA_ARGS=()
# We ship only the runtime libs — never build the test executables. (--skip_tests
# only skips RUNNING them; this skips building them, fixing test-only link errors
# like RegisterCustomOpsAltName and cutting a large chunk of build time.)
EXTRA_ARGS+=(--cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF)
[ "${ORT_ENABLE_LTO:-1}" = "1" ] && EXTRA_ARGS+=(--enable_lto)
if [ -n "${ORT_CUDA_ARCHS:-}" ] && { [[ "${ORT_EP_FLAGS}" == *use_cuda* ]] || [[ "${ORT_EP_FLAGS}" == *nv_tensorrt_rtx* ]]; }; then
   EXTRA_ARGS+=(--cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${ORT_CUDA_ARCHS}")
fi
# nvcc parallelizes the multi-arch compile per file (opt-in; helps the CUDA groups).
if [ -n "${NVCC_THREADS:-}" ] && { [[ "${ORT_EP_FLAGS}" == *use_cuda* ]] || [[ "${ORT_EP_FLAGS}" == *nv_tensorrt_rtx* ]]; }; then
   EXTRA_ARGS+=(--nvcc_threads "${NVCC_THREADS}")
fi
# AMD (MIGraphX): disable Composable Kernel (fused attention/GEMM — unused by
# CNN/YOLO inference, an enormous build, and it errors with empty
# HIP_ARCHITECTURES on the composable_kernel_fmha target). Set the gfx arch list
# for the HIP targets (runtime coverage bounded by the MIOpen/rocBLAS in
# ROCM_IMAGE). Keyed off use_migraphx since --use_rocm no longer exists in 1.23.
if [[ "${ORT_EP_FLAGS}" == *use_migraphx* ]]; then
   EXTRA_ARGS+=(--cmake_extra_defines onnxruntime_USE_COMPOSABLE_KERNEL=OFF)
   [ -n "${ORT_ROCM_GFX:-}" ] && EXTRA_ARGS+=(--cmake_extra_defines "CMAKE_HIP_ARCHITECTURES=${ORT_ROCM_GFX}")
fi

# clang (used for the WebGPU build) rejects ORT's GCC-style 2-arg
# __builtin_ia32_tpause in spin_pause.cc — clang's builtin takes 3 args. Swap it
# for the portable _tpause() intrinsic, which both GCC and clang accept with the
# (control, uint64 counter) signature the file's own _WIN32 branch already uses.
if [[ "${ORT_EP_FLAGS}" == *use_webgpu* ]]; then
   sp="${SRC_DIR}/onnxruntime/core/common/spin_pause.cc"
   if [ -f "$sp" ] && grep -q '__builtin_ia32_tpause' "$sp"; then
      sed -i 's/__builtin_ia32_tpause(/_tpause(/g' "$sp"
      echo "Patched spin_pause.cc: __builtin_ia32_tpause -> _tpause (clang compat)"
   fi
fi

# Dawn (WebGPU) is a clang-only codebase: with GCC it fails to compile
# (`redefinition of class dawn::native::stream::Stream<T>`) and its build passes
# clang-only `-Wno-*` flags GCC rejects. So when building WebGPU, compile ORT's
# host C++ with clang. Prefer ROCm's bundled clang (modern, and libstdc++
# ABI-compatible with the gcc-built ROCm/MIGraphX libs we link); fall back to any
# system clang. HIP kernels still go through ORT's own HIP toolchain detection.
#
# Pass the compiler EXPLICITLY to cmake (env CC/CXX is ignored once a build dir
# has a cached compiler — e.g. a persistent BUILD_STORAGE_DIR from a prior GCC
# run), and wipe the build dir if it was configured with a different compiler
# (cmake refuses to switch compilers in place, and GCC/clang objects can't be
# LTO-linked together anyway).
if [[ "${ORT_EP_FLAGS}" == *use_webgpu* ]]; then
   WEBGPU_CXX=""
   for cxx in /opt/rocm/llvm/bin/clang++ clang++-18 clang++-17 clang++-16 clang++-15 clang++; do
      if command -v "$cxx" >/dev/null 2>&1 || [ -x "$cxx" ]; then WEBGPU_CXX="$cxx"; break; fi
   done
   [ -n "${WEBGPU_CXX}" ] || { echo "ERROR: WebGPU build needs clang; none found." >&2; exit 1; }
   WEBGPU_CC="${WEBGPU_CXX/clang++/clang}"
   export CC="${WEBGPU_CC}" CXX="${WEBGPU_CXX}"
   EXTRA_ARGS+=(--cmake_extra_defines "CMAKE_C_COMPILER=${WEBGPU_CC}"
                --cmake_extra_defines "CMAKE_CXX_COMPILER=${WEBGPU_CXX}"
                # Build Dawn as a monolithic shared lib (libwebgpu_dawn.so) that
                # EXPORTS the WebGPU C API, with libonnxruntime dynamically
                # linking it. Lets our Rust side FFI the SAME Dawn ORT uses →
                # inject a shared device + bind a WGPUBuffer as a zero-copy
                # input. (Default OFF statically archives Dawn, 0 exported wgpu*
                # symbols.) Do NOT also set USE_EXTERNAL_DAWN — mutually
                # exclusive, and it forces us to ship the proc table instead.
                --cmake_extra_defines onnxruntime_BUILD_DAWN_MONOLITHIC_LIBRARY=ON)
   echo "WebGPU build -> host compiler: ${WEBGPU_CXX} ($("${WEBGPU_CXX}" --version 2>/dev/null | head -1))"
   cache="${BUILD_DIR}/Release/CMakeCache.txt"
   if [ -f "$cache" ] && ! grep -qsF "CMAKE_CXX_COMPILER:FILEPATH=${WEBGPU_CXX}" "$cache"; then
      echo "Build dir was configured with a different compiler; wiping ${BUILD_DIR} for a clean clang configure."
      rm -rf "${BUILD_DIR}"
   fi
fi

cd "${SRC_DIR}"
# shellcheck disable=SC2086
python3 tools/ci_build/build.py \
   --build_dir "${BUILD_DIR}" \
   --config Release \
   --parallel "${NPROC}" \
   --skip_tests \
   --build_shared_lib \
   --allow_running_as_root \
   --compile_no_warning_as_error \
   --cmake_generator Ninja \
   ${ORT_EP_FLAGS} \
   ${ORT_EXTRA_FLAGS:-} \
   "${EXTRA_ARGS[@]}"

REL="${BUILD_DIR}/Release"
mkdir -p "${OUT_DIR}/lib"
find "${REL}" -maxdepth 1 -name 'libonnxruntime*.so*' -exec cp -a {} "${OUT_DIR}/lib/" \;

# WebGPU: depending on ORT's cmake, Dawn is either statically linked into
# libonnxruntime (nothing extra to harvest) or emitted as a separate
# libwebgpu_dawn.so that libonnxruntime needs at load time. Harvest it if
# present — it may live in a _deps subdir, not next to libonnxruntime in
# Release/, so search the whole build tree. (Downstream: this .so must be
# staged + bundled into the AppImage on the app side.)
if [[ "${ORT_EP_FLAGS}" == *use_webgpu* ]]; then
   find "${BUILD_DIR}" -name 'libwebgpu_dawn*.so*' -exec cp -a {} "${OUT_DIR}/lib/" \; 2>/dev/null || true

   # Harvest the GENERATED Dawn C header that matches THIS libwebgpu_dawn.so.
   # The app FFIs Dawn via bindgen, so its struct layouts must come from this
   # exact header — a hand-vendored copy from a different Dawn version leaves
   # newer trailing struct fields uninitialized and ORT segfaults reading past
   # them. Several webgpu.h exist in the tree (vanilla webgpu-headers, etc.);
   # pick the Dawn-flavored one, identified by a Dawn-only symbol.
   dawn_hdr=""
   while IFS= read -r cand; do
      if grep -q 'WGPUSharedTextureMemory' "$cand" 2>/dev/null; then dawn_hdr="$cand"; break; fi
   done < <(find "${BUILD_DIR}" -name 'webgpu.h' 2>/dev/null)
   if [ -n "$dawn_hdr" ]; then
      mkdir -p "${OUT_DIR}/include/dawn"
      cp -a "$dawn_hdr" "${OUT_DIR}/include/dawn/webgpu.h"
      echo "Harvested Dawn header: ${dawn_hdr} -> ${OUT_DIR}/include/dawn/webgpu.h"
   else
      echo "WARN: use_webgpu build but no Dawn-flavored webgpu.h found under ${BUILD_DIR}" >&2
   fi
fi

{
   echo "onnxruntime ${ORT_VERSION}  group=${ORT_GROUP}  arch=$(uname -m)"
   echo "EP flags: ${ORT_EP_FLAGS} ${ORT_EXTRA_FLAGS:-}"
   echo "LTO=${ORT_ENABLE_LTO:-1} CUDA=${ORT_CUDA_ARCHS:-n/a} GFX=${ORT_ROCM_GFX:-n/a}"
   echo
   for so in "${OUT_DIR}"/lib/*.so*; do
      [ -e "$so" ] || continue
      soname=$(readelf -d "$so" 2>/dev/null | awk -F'[][]' '/SONAME/{print $2}')
      printf '  %-50s soname=%-42s %10s bytes\n' \
         "$(basename "$so")" "${soname:-—}" "$(stat -c%s "$so")"
   done
   echo
   ( cd "${OUT_DIR}/lib" && sha256sum -- *.so* )
} | tee "${OUT_DIR}/MANIFEST.txt"

echo "Done: ${OUT_DIR}/"
