#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_dir/images"

# Official Google Drive file ID for the current Ubuntu 22.04 Server image.
file_id='1tJdNc71VrvljfnE56tCYxOsBvB-olo43'
output="$project_dir/images/Orangepizero3w_1.0.2_ubuntu_jammy_server_linux6.6.98.7z"
url="https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"

echo "Retrying official download: $output"
wget -c --show-progress -O "$output" "$url"
if file "$output" | grep -qi 'HTML'; then
  echo 'The server returned an HTML error page (often Google Drive quota exceeded); remove it and retry later.' >&2
  exit 2
fi
7z t "$output"
