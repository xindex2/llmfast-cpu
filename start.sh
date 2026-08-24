#!/usr/bin/env bash
# Start the gateway in the foreground (containers / RunPod / anywhere without systemd).
cd "$(dirname "$0")"
set -a; source ./gateway.env; set +a
exec ./gateway/llmfast-gateway
