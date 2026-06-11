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
   export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi

EXTRA_ARGS=()
[ "${ORT_ENABLE_LTO:-1}" = "1" ] && EXTRA_ARGS+=(--enable_lto)
if [ -n "${ORT_CUDA_ARCHS:-}" ] && { [[ "${ORT_EP_FLAGS}" == *use_cuda* ]] || [[ "${ORT_EP_FLAGS}" == *nv_tensorrt_rtx* ]]; }; then
   EXTRA_ARGS+=(--cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${ORT_CUDA_ARCHS}")
fi
# Runtime gfx coverage is bounded by the MIOpen/rocBLAS in ROCM_IMAGE.
if [ -n "${ORT_ROCM_GFX:-}" ] && [[ "${ORT_EP_FLAGS}" == *use_rocm* ]]; then
   EXTRA_ARGS+=(--cmake_extra_defines "CMAKE_HIP_ARCHITECTURES=${ORT_ROCM_GFX}")
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
