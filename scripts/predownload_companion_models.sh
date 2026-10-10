#!/usr/bin/env bash
# Host-only, finite download job. No board access or inference.
set -u
cd -- "$(dirname -- "$0")/.." || exit 1
mkdir -p outputs/model-downloads
exec 9>outputs/model-downloads/job.lock
flock -n 9 || exit 1

download_model() {
    local model="$1" revision="$2" attempt
    for attempt in {1..12}; do
        printf '\nAttempt %s at %s\n' "$attempt" "$(date -Is)"
        if python3 -u scripts/download_smolvlm.py \
            --manifest configs/vlm-models.json --model "$model" \
            --directory "models/$model" --workers 4 \
            --mirror https://modelscope.cn/models \
            --source-revision "$revision" --no-proxy '*'; then
            printf 'COMPLETE %s %s\n' "$model" "$(date -Is)"
            return 0
        fi
        sleep 30
    done
    printf 'FAILED %s; partial chunks retained\n' "$model"
    return 1
}

download_model qwen3-17b-q4 1e3f49488e445e2148d2e065ea0612c320da65c3 >outputs/model-downloads/qwen3-17b-q4.log 2>&1 &
first_pid=$!
download_model qwen25-15b-q4 e8b19c78f775ccbcf6df15ceace6bb5276f09765 >outputs/model-downloads/qwen25-15b-q4.log 2>&1 &
second_pid=$!
result=0
wait "$first_pid" || result=1
wait "$second_pid" || result=1
printf 'JOB_EXIT=%s %s\n' "$result" "$(date -Is)"
exit "$result"
