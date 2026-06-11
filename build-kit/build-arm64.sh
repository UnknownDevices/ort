#!/usr/bin/env bash
# Build aarch64 EP groups on the NVIDIA DGX: cpu, cuda-trt (native).
# No nv-trt-rtx (consumer RTX is x86_64 only). Artifacts -> ./out/<group>/lib/.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=VERSIONS.env
source ./VERSIONS.env

: "${ORT_CUDA_ARCHS:=72-real;87-real;90-real;90-virtual}"

DOCKER="${DOCKER:-docker}"
mkdir -p out sdk

RUN_FLAGS=(--rm -v "$PWD/out:/out"
   -e NPROC="${NPROC:-$(nproc)}"
   -e NVCC_THREADS="${NVCC_THREADS:-}"
   -e ORT_ENABLE_LTO="${ORT_ENABLE_LTO:-1}"
   -e ORT_CUDA_ARCHS="${ORT_CUDA_ARCHS}")
[ -n "${ORT_EXTRA_FLAGS:-}" ] && RUN_FLAGS+=(-e ORT_EXTRA_FLAGS="${ORT_EXTRA_FLAGS}")
if [ -n "${CCACHE_DIR_HOST:-}" ]; then
   mkdir -p "${CCACHE_DIR_HOST}"
   RUN_FLAGS+=(-v "$(realpath "${CCACHE_DIR_HOST}"):/ccache" -e CCACHE_DIR=/ccache)
fi

build_group() {
   local group="$1" dockerfile="$2"; shift 2
   echo; echo "### building ${group} (aarch64) ###"
   "$DOCKER" build -f "docker/${dockerfile}" \
      --build-arg ORT_VERSION="${ORT_VERSION}" "$@" \
      -t "ort-build-${group}:${ORT_VERSION}" .
   "$DOCKER" run "${RUN_FLAGS[@]}" "ort-build-${group}:${ORT_VERSION}"
}

build_group cpu      Dockerfile.cpu      --build-arg CPU_IMAGE="${CPU_IMAGE}"
build_group cuda-trt Dockerfile.cuda-trt --build-arg NV_TENSORRT_IMAGE="${NV_TENSORRT_IMAGE}"

echo; echo "arm64 builds complete -> ./out/"
./check_outputs.sh
