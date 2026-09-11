#!/usr/bin/env bash
# Project-local Linux x86_64 tools. Requires curl, tar, cmake, a C++ compiler.
set -euo pipefail
drugforge_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
drugforge_tools="$drugforge_root/.tools"
drugforge_build="$(mktemp -d)"
trap 'rm -rf "$drugforge_build"' EXIT
mkdir -p "$drugforge_tools/bin" "$drugforge_tools/java25"
curl -fL --retry 2 https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 -o "$drugforge_tools/bin/vina"
chmod +x "$drugforge_tools/bin/vina"
curl -fL --retry 2 https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz -o "$drugforge_build/p2rank.tar.gz"
tar -xzf "$drugforge_build/p2rank.tar.gz" -C "$drugforge_tools"
curl -fL --retry 2 'https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jre/hotspot/normal/eclipse' -o "$drugforge_build/java.tar.gz"
tar -xzf "$drugforge_build/java.tar.gz" -C "$drugforge_tools/java25" --strip-components=1
curl -fL --retry 2 https://github.com/openbabel/openbabel/archive/refs/tags/openbabel-3-2-1.tar.gz -o "$drugforge_build/ob.tar.gz"
tar -xzf "$drugforge_build/ob.tar.gz" -C "$drugforge_build"
cmake -S "$drugforge_build/openbabel-openbabel-3-2-1" -B "$drugforge_build/build" -DCMAKE_INSTALL_PREFIX="$drugforge_tools" -DBUILD_GUI=OFF -DENABLE_TESTS=OFF
cmake --build "$drugforge_build/build" -j 4
cmake --install "$drugforge_build/build"
"$drugforge_tools/bin/vina" --version
"$drugforge_tools/bin/obabel" -V
"$drugforge_tools/java25/bin/java" -version
