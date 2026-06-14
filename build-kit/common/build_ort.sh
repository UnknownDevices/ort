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
