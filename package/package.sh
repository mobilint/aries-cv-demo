#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
   echo "Usage: $0 <PRODUCT> <DRIVER_TYPE>" >&2
   exit 1
fi
readonly PRODUCT="$1"
readonly DRIVER_TYPE="$2"

readonly PACKAGE_DIR="demo-package-${PRODUCT}"
readonly ROOT_DIR="$(git rev-parse --show-toplevel)"

copy_first_existing() {
    local dest="$1"
    shift
    for candidate in "$@"; do
        if [[ -e "$candidate" ]]; then
            cp -r "$candidate" "$dest"
            return 0
        fi
    done
    return 1
}

cd "$ROOT_DIR"
rm -rf "$PACKAGE_DIR" "$PACKAGE_DIR.tar.gz"

mkdir -p "$PACKAGE_DIR/yaml-cpp/lib"
mkdir -p "$PACKAGE_DIR/qbruntime/lib"

cp -r assets src package/Makefile package/README.md run.sh "$PACKAGE_DIR"

mkdir -p "$PACKAGE_DIR/yaml-cpp/include"
copy_first_existing "$PACKAGE_DIR/yaml-cpp/include" \
    /usr/include/yaml-cpp \
    /usr/local/include/yaml-cpp

copy_first_existing "$PACKAGE_DIR/yaml-cpp/lib/" \
    /usr/lib/x86_64-linux-gnu/libyaml-cpp.so \
    /usr/local/lib/libyaml-cpp.so

mkdir -p "$PACKAGE_DIR/qbruntime/include"
copy_first_existing "$PACKAGE_DIR/qbruntime/include" \
    /usr/include/qbruntime \
    /usr/local/include/qbruntime

copy_first_existing "$PACKAGE_DIR/qbruntime/lib/" \
    /usr/lib/x86_64-linux-gnu/libqbruntime.so \
    /usr/local/lib/libqbruntime.so

tar -czvf "$PACKAGE_DIR.tar.gz" "$PACKAGE_DIR"
rm -rf "$PACKAGE_DIR"
