#!/usr/bin/env bash
# Verify each built group harvested libonnxruntime.so + its provider .so's.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d out ] || [ -z "$(ls -A out 2>/dev/null)" ]; then
   echo "No artifacts in ./out yet."
   exit 0
fi

declare -A EXPECT=(
   [cpu]=""
   [webgpu]=""
   [rocm]="libonnxruntime_providers_rocm.so libonnxruntime_providers_migraphx.so libonnxruntime_providers_shared.so"
   [openvino]="libonnxruntime_providers_openvino.so libonnxruntime_providers_shared.so"
   [cuda-trt]="libonnxruntime_providers_cuda.so libonnxruntime_providers_tensorrt.so libonnxruntime_providers_shared.so"
   [nv-trt-rtx]="libonnxruntime_providers_nv_tensorrt_rtx.so libonnxruntime_providers_shared.so"
)

rc=0
for d in out/*/; do
   g="$(basename "$d")"
   echo "== ${g} =="
   libdir="${d}lib"
   if [ ! -d "$libdir" ]; then echo "  ! no lib/ dir"; rc=1; continue; fi

   shopt -s nullglob
   for so in "$libdir"/*.so*; do printf '  %s\n' "$(basename "$so")"; done
   shopt -u nullglob

   if [ ! -e "$libdir/libonnxruntime.so" ] && ! ls "$libdir"/libonnxruntime.so.* >/dev/null 2>&1; then
      echo "  ! MISSING libonnxruntime.so"; rc=1
   fi
   for want in ${EXPECT[$g]:-}; do
      [ -e "$libdir/$want" ] || { echo "  ! MISSING ${want}"; rc=1; }
   done
   echo
done

[ "$rc" -eq 0 ] && echo "OK" || echo "INCOMPLETE"
exit "$rc"
