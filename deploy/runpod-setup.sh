#!/bin/bash
# One-shot RunPod GPU pod setup / recovery. Safe to re-run after any container reset:
#   curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/deploy/runpod-setup.sh | bash
#
# Assumes the pod was created with: env NVIDIA_DRIVER_CAPABILITIES=all, HTTP port 8080
# exposed, and a /workspace volume big enough for the models (see docs/install-gpu-runpod.md).

set -u

echo "== packages"
apt-get update -qq && apt-get install -y -qq libvulkan1 vulkan-tools curl git >/dev/null

echo "== vulkan ICD manifest (RunPod mounts the driver libraries but not this file)"
mkdir -p /usr/share/vulkan/icd.d
printf '{ "file_format_version": "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.277" } }\n' > /usr/share/vulkan/icd.d/nvidia_icd.json

echo "== driver userspace libraries the mount omits (SPIR-V compiler etc.)"
V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
RUN=/workspace/NVIDIA-Linux-x86_64-$V.run
# -s, not -f: a failed curl leaves a 0-byte file behind, and existence-checking it means every
# later run "extracts" an empty archive and silently fixes nothing. Size is the health check.
if [ ! -s "$RUN" ]; then
  rm -f "$RUN"
  curl -fLo "$RUN" "https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run" \
    || curl -fLo "$RUN" "https://download.nvidia.com/XFree86/Linux-x86_64/$V/NVIDIA-Linux-x86_64-$V.run" \
    || { rm -f "$RUN"; echo "!! driver download failed for $V — vulkan fix cannot proceed"; }
fi
[ -s "$RUN" ] && echo "   installer: $(du -h "$RUN" | cut -f1)"
rm -rf /tmp/nvd
sh "$RUN" --extract-only --target /tmp/nvd >/dev/null || true
added=0
for f in /tmp/nvd/*.so."$V"; do
  b=$(basename "$f")
  [ -e "/usr/lib/x86_64-linux-gnu/$b" ] || { cp "$f" /usr/lib/x86_64-linux-gnu/; added=$((added+1)); }
done
ldconfig
echo "   added $added libraries at $V"
vulkaninfo --summary 2>/dev/null | grep deviceName || echo "!! vulkan still broken — paste the errors from: vulkaninfo --summary"

echo "== llmfast"
[ -d /opt/llmfast ] || curl -fsSL https://raw.githubusercontent.com/xindex2/llmfast-cpu/main/install.sh | bash
cd /opt/llmfast
mkdir -p /workspace/models
grep -q MODELS_DIR gateway.env 2>/dev/null || echo "MODELS_DIR=/workspace/models" >> gateway.env
grep -q GPU_MEM_MB gateway.env 2>/dev/null || echo "GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" >> gateway.env

echo "== gateway"
pkill -f llmfast-gateway 2>/dev/null; sleep 1
nohup ./start.sh > gateway.log 2>&1 &
sleep 2
curl -s -o /dev/null -w "   gateway on 8080: HTTP %{http_code}\n" http://localhost:8080/admin/ui
grep ADMIN_TOKEN gateway.env
echo "== done — admin UI: https://<pod-id>-8080.proxy.runpod.net/admin/ui"
