#!/usr/bin/env bash
# Build ONLY the AMD group (x86_64): MIGraphX + WebGPU (Dawn/Vulkan) in one
# libonnxruntime core. Same env/flags as build-amd64.sh, just this one group.
# Artifacts -> ./out/rocm/lib/.
#
# Heavy build (Dawn compiles from source). Optional env:
#   BUILD_STORAGE_DIR=/path   mount over the container /tmp (avoids ENOSPC on a
#                             small docker data-root; the build tree is large)
#   CCACHE_DIR_HOST=/path     persistent ccache across re-runs
#   NPROC=N                   build parallelism (default: nproc)
#   ORT_ROCM_GFX="gfx...;..."  gfx arch list for the HIP/MIGraphX targets
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=VERSIONS.env
source ./VERSIONS.env

: "${ORT_ROCM_GFX:=gfx906;gfx1010;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201}"

DOCKER="${DOCKER:-docker}"
mkdir -p out

RUN_FLAGS=(--rm -v "$PWD/out:/out"
   -e NPROC="${NPROC:-$(nproc)}"
   -e ORT_ENABLE_LTO="${ORT_ENABLE_LTO:-1}"
   -e ORT_ROCM_GFX="${ORT_ROCM_GFX}")
[ -n "${ORT_EXTRA_FLAGS:-}" ] && RUN_FLAGS+=(-e ORT_EXTRA_FLAGS="${ORT_EXTRA_FLAGS}")
if [ -n "${CCACHE_DIR_HOST:-}" ]; then
   mkdir -p "${CCACHE_DIR_HOST}"
   RUN_FLAGS+=(-v "$(realpath "${CCACHE_DIR_HOST}"):/ccache" -e CCACHE_DIR=/ccache)
fi
if [ -n "${BUILD_STORAGE_DIR:-}" ]; then
   mkdir -p "${BUILD_STORAGE_DIR}"
   RUN_FLAGS+=(-v "$(realpath "${BUILD_STORAGE_DIR}"):/tmp")
fi

echo "### building rocm (MIGraphX + WebGPU) | ORT ${ORT_VERSION} | gfx=${ORT_ROCM_GFX} ###"
"$DOCKER" build -f docker/Dockerfile.rocm \
   --build-arg ORT_VERSION="${ORT_VERSION}" \
   --build-arg ROCM_IMAGE="${ROCM_IMAGE}" \
   -t "ort-build-rocm:${ORT_VERSION}" .
"$DOCKER" run "${RUN_FLAGS[@]}" "ort-build-rocm:${ORT_VERSION}"

echo; echo "rocm build complete -> ./out/rocm/"
echo "--- harvested libs ---"; ls -1 out/rocm/lib/ 2>/dev/null || true
echo "--- WebGPU in the core? ---"
real_so="$(ls out/rocm/lib/libonnxruntime.so.*.* 2>/dev/null | head -1)"
strings -a "${real_so:-/dev/null}" 2>/dev/null | grep -qi "onnxruntime::webgpu\|WebGpuContext\|WebGpuProviderFactory" \
   && echo "  webgpu EP present in ${real_so}" || echo "  WARNING: no webgpu EP symbols found"
echo "--- separate Dawn lib? (else statically linked) ---"
ls -1 out/rocm/lib/ 2>/dev/null | grep -i dawn || echo "  none (Dawn is static in libonnxruntime)"