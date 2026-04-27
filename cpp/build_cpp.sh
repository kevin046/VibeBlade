#!/usr/bin/env bash
# Build VibeBlade native C++ backend
# Usage: ./build_cpp.sh [Release|Debug]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${1:-Release}"
PYTHON="$(python3 -c 'import sys; print(sys.executable)')"

echo "╔══════════════════════════════════════════════╗"
echo "║  VibeBlade Native Backend Build            ║"
echo "╚══════════════════════════════════════════════╝"
echo "  Build type:  ${BUILD_TYPE}"
echo "  Python:      ${PYTHON}"
echo "  CXX:         $(g++ --version | head -1)"
echo ""

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
PYBIND_DIR=$(python3 -c "import pybind11, os; print(os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11'))")
echo "▶ Configuring..."
cmake .. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DPython3_EXECUTABLE="${PYTHON}" \
    -DPYBIND11_FINDPYTHON=ON \
    -Dpybind11_DIR="${PYBIND_DIR}" \
    2>&1 | { grep -v '^-- ' || true; }

# Build
echo ""
echo "▶ Building..."
cmake --build . -j"$(nproc)" 2>&1

# Copy .so into package
SO_FILE=$(find . -name "_vibeblade_native*.so" -o -name "_vibeblade_native*.pyd" | head -1)
if [ -z "${SO_FILE}" ]; then
    echo "✗ Build failed — no .so/.pyd found"
    exit 1
fi

DEST="${SCRIPT_DIR}/../vibeblade/_vibeblade_native$(echo ${SO_FILE} | grep -oP '\.(so|pyd)$')"
cp "${SO_FILE}" "${DEST}"
echo ""
echo "✓ Copied ${SO_FILE} → ${DEST}"

# Verify import
echo ""
echo "▶ Verifying import..."
cd "${SCRIPT_DIR}/.."
if python3 -c "
import vibeblade._vibeblade_native as nat
print(f'  SIMD backend: {nat.SIMD_BACKEND}')
print(f'  Functions: {sorted([x for x in dir(nat) if not x.startswith(\"_\")])}')
"; then
    echo ""
    echo "✓ Native backend ready!"
else
    echo ""
    echo "✗ Import failed — check above errors"
    exit 1
fi
