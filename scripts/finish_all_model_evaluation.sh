#!/usr/bin/env bash
# Board-only continuation. Wait for the running history matrix before taking its lock.
set -euo pipefail
cd /opt/opi-vlm
while systemctl is-active --quiet opi-history-matrix-20261010.service; do
    sleep 30
done
python3 -c 'import json; r=json.load(open("history-matrix-20261010.json")); assert r["status"] in ("COMPLETE", "COMPLETE_WITH_ERRORS") and r.get("production_restored"), "History run did not finish safely"'
python3 -u evaluate_companion.py --plan companion-all-models-plan.json \
    --output companion-other-four-20261010.json \
    --models smolvlm-256m smolvlm-500m internvl3-1b qwen25vl-3b-q4
