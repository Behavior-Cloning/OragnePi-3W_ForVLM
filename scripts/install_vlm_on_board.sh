#!/usr/bin/env bash
# Run on the Orange Pi as root from a copied project checkout.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
source_revision=d81235049384534c167caea52b85a694f6103d14
if [[ $(id -u) != 0 || $(uname -m) != aarch64 ]]; then
    echo 'Run as root on the aarch64 Orange Pi board.' >&2
    exit 1
fi
if ! tr '\0' '\n' < /proc/device-tree/compatible | grep -q 'sun60i'; then
    echo 'Expected an A733/sun60i board; refusing machine-specific CPU configuration.' >&2
    exit 1
fi
cd "$project_dir/models/smolvlm-256m"
sha256sum -c <<'CHECKSUMS'
2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65  SmolVLM-256M-Instruct-Q8_0.gguf
0802360aca1748f112ea510b8ff277c65b1361c8ef30ed89b83c9c7a60d08e96  mmproj-SmolVLM-256M-Instruct-f16.gguf
CHECKSUMS
apt-get update
apt-get install -y build-essential cmake ca-certificates curl python3
install -d /opt/opi-vlm/source /opt/opi-vlm/bin /opt/opi-vlm/models /opt/opi-vlm/app
archive="$project_dir/data/llama-$source_revision.tar.gz"
if [[ ! -f "$archive" ]]; then
    install -d "$project_dir/data"
    curl -fL --retry 3 -o "$archive.part" "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$source_revision"
    mv "$archive.part" "$archive"
fi
printf '%s  %s\n' 6b58785f0a82898f4c3442417ff962e2d4b231b1bee5d033902ad90b27901e14 "$archive" | sha256sum -c -
tar -xzf "$archive" -C /opt/opi-vlm/source --strip-components=1
cmake -S /opt/opi-vlm/source -B /opt/opi-vlm/build \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF
# Limit parallel compilation to reduce heat and memory use.
cmake --build /opt/opi-vlm/build --target llama-server -j2
install -m 755 /opt/opi-vlm/build/bin/llama-server /opt/opi-vlm/bin/
install -m 644 "$project_dir"/models/smolvlm-256m/*.gguf /opt/opi-vlm/models/
cp -a "$project_dir/src/vlm_chat/." /opt/opi-vlm/app/
if ! id opi-vlm >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin opi-vlm
fi
install -m 644 "$project_dir"/configs/systemd/opi-vlm-*.service /etc/systemd/system/
install -m 644 "$project_dir/configs/vlm-models.json" /opt/opi-vlm/vlm-models.json
install -d /etc/systemd/system/opi-vlm-model.service.d /etc/systemd/system/opi-vlm-web.service.d
install -m 644 "$project_dir/configs/systemd/web-control.conf" /etc/systemd/system/opi-vlm-web.service.d/model.conf
systemctl daemon-reload
systemctl enable --now opi-vlm-model.service opi-vlm-control.service opi-vlm-web.service
systemctl restart opi-vlm-model.service opi-vlm-web.service
echo 'Installed. Verify: curl http://127.0.0.1:8080/api/health'
echo 'Open http://BOARD_IP:8080 from a device on the same LAN.'
