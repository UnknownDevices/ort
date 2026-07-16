#!/usr/bin/env bash
# Build every x86_64 EP group: cpu(+oneDNN), rocm(+migraphx+webgpu), openvino,
# cuda-trt, nv-trt-rtx. Artifacts -> ./out/<group>/lib/. (WebGPU/Dawn is folded
# into the rocm core; the standalone webgpu group is disabled by default below.)
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=VERSIONS.env
source ./VERSIONS.env

: "${ORT_CUDA_ARCHS:=75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual}"
: "${ORT_ROCM_GFX:=gfx906;gfx1010;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201}"

DOCKER="${DOCKER:-docker}"
mkdir -p out

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
# Put the large build trees + compiler temp files on a roomier disk than Docker's data-root:
# BUILD_STORAGE_DIR is mounted over the container's /tmp, so BUILD_DIR (/tmp/ort-build) and
# gcc's temp .s files land there. Avoids ENOSPC on the cuda groups when /var/lib/docker is small.
if [ -n "${BUILD_STORAGE_DIR:-}" ]; then
   mkdir -p "${BUILD_STORAGE_DIR}"
   RUN_FLAGS+=(-v "$(realpath "${BUILD_STORAGE_DIR}"):/tmp")
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

# AMD: MIGraphX (ROCm 7.x, RDNA2..RDNA4) + the WebGPU EP (Dawn/Vulkan) built
# into the SAME core as a driver-portable second provider — the app selects at
# runtime. See docker/Dockerfile.rocm. This build is heavy (Dawn compiles from
# source); ensure enough disk (BUILD_STORAGE_DIR) and expect a long run.
build_group rocm       Dockerfile.rocm       --build-arg ROCM_IMAGE="${ROCM_IMAGE}"
build_group openvino   Dockerfile.openvino   --build-arg OPENVINO_IMAGE="${OPENVINO_IMAGE}"

# NVIDIA groups: device-LTO off + bounded parallelism (see VERSIONS.env). These
# -e flags follow RUN_FLAGS on the `docker run` line, so they override the globals.
EXTRA_RUN_ENV=(-e ORT_ENABLE_LTO="${CUDA_ENABLE_LTO}"
   -e NPROC="${CUDA_NPROC}" -e NVCC_THREADS="${CUDA_NVCC_THREADS}")
build_group cuda-trt   Dockerfile.cuda-trt   --build-arg NV_TENSORRT_IMAGE="${NV_TENSORRT_IMAGE}"
build_group nv-trt-rtx Dockerfile.nv-trt-rtx --build-arg NV_RTX_CUDA_IMAGE="${NV_RTX_CUDA_IMAGE}" --build-arg TENSORRT_RTX_VERSION="${TENSORRT_RTX_VERSION}"
EXTRA_RUN_ENV=()

# The WebGPU EP now ships inside the `rocm` group above (same libonnxruntime
# core), so the AMD bundle gets it from there. This standalone group is left
# for a WebGPU-only / other-vendor build but is redundant for the AMD app —
# disabled by default to avoid a second (heavy) Dawn compile. Re-enable if you
# need a MIGraphX-free WebGPU core.
# build_group webgpu     Dockerfile.webgpu     --build-arg WEBGPU_IMAGE="${CPU_IMAGE}"

echo; echo "amd64 builds complete -> ./out/"
./check_outputs.sh
