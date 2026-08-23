#!/usr/bin/env bash
# Runs engine + gateway + admin UI for local development.
set -e
cd "$(dirname "$0")"
MODEL="${MODEL:-../models/qwen3-0.6b}"
(cd engine && MODEL="$MODEL" ./target/release/forge-engine) &
sleep 3
(cd gateway && ./forge-gateway) &
(cd admin && npm run dev) &
wait
