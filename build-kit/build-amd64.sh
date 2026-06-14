#!/usr/bin/env bash
# Build every x86_64 EP group: cpu(+oneDNN), rocm(+migraphx), openvino,
# cuda-trt, nv-trt-rtx, webgpu. Artifacts -> ./out/<group>/lib/.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=VERSIONS.env
source ./VERSIONS.env

: "${ORT_CUDA_ARCHS:=75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual}"
: "${ORT_ROCM_GFX:=gfx906;gfx1010;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201}"

DOCKER="${DOCKER:-docker}"
mkdir -p out sdk

RUN_FLAGS=(--rm -v "$PWD/out:/out"
   -e NPROC="${NPROC:-$(nproc)}"
   -e NVCC_THREADS="${NVCC_THREADS:-}"
   -e ORT_ENABLE_LTO="${ORT_ENABLE_LTO:-1}"
   -e ORT_CUDA_ARCHS="${ORT_CUDA_ARCHS}"
   -e ORT_ROCM_GFX="${ORT_ROCM_GFX}")
[ -n "${ORT_EXTRA_FLAGS:-}" ] && RUN_FLAGS+=(-e ORT_EXTRA_FLAGS="${ORT_EXTRA_FLAGS}")
if [ -n "${CCACHE_DIR_HOST:-}" ]; then
   mkdir -p "${CCACHE_DIR_HOST}"
   RUN_FLAGS+=(-v "$(realpath "${CCACHE_DIR_HOST}"):/ccache" -e CCACHE_DIR=/ccache)
fi

EXTRA_RUN_ENV=()
build_group() {
   local group="$1" dockerfile="$2"; shift 2
   echo; echo "### building ${group} ###"
   "$DOCKER" build -f "docker/${dockerfile}" \
      --build-arg ORT_VERSION="${ORT_VERSION}" "$@" \
      -t "ort-build-${group}:${ORT_VERSION}" .
   "$DOCKER" run "${RUN_FLAGS[@]}" "${EXTRA_RUN_ENV[@]}" "ort-build-${group}:${ORT_VERSION}"
}

# oneDNN is x86-only; arm64 cpu keeps XNNPACK + KleidiAI.
EXTRA_RUN_ENV=(-e ORT_EP_FLAGS="--use_xnnpack --use_dnnl")
build_group cpu        Dockerfile.cpu        --build-arg CPU_IMAGE="${CPU_IMAGE}"
EXTRA_RUN_ENV=()

build_group rocm       Dockerfile.rocm       --build-arg ROCM_IMAGE="${ROCM_IMAGE}"
build_group openvino   Dockerfile.openvino   --build-arg OPENVINO_IMAGE="${OPENVINO_IMAGE}"
build_group cuda-trt   Dockerfile.cuda-trt   --build-arg NV_TENSORRT_IMAGE="${NV_TENSORRT_IMAGE}"
build_group nv-trt-rtx Dockerfile.nv-trt-rtx \
   --build-arg NV_TENSORRT_IMAGE="${NV_TENSORRT_IMAGE}" \
   --build-arg TRT_RTX_URL="${TRT_RTX_URL}"
build_group webgpu     Dockerfile.webgpu     --build-arg WEBGPU_IMAGE="${CPU_IMAGE}"

echo; echo "amd64 builds complete -> ./out/"
./check_outputs.sh
