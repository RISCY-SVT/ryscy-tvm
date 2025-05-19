#!/usr/bin/env bash
set -eEo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

########################    PARAMETERS   ##########################
source ./env.sh
###################################################################

# Error handling
function handle_error {
    echo "ERROR: Build failed at line $1"
    exit 1
}
trap 'handle_error $LINENO' ERR

echo
echo "=================>>> 0. System packages"
sudo apt-get update
sudo apt-get install -y git build-essential libstdc++-12-dev cmake ninja-build python3.12-venv \
                        libedit-dev libxml2-dev zlib1g-dev libz-dev libzstd-dev

echo
echo "=================>>> 1. Environment setup"
# Check LLVM installation from our custom build
if [ ! -f "$TOOLROOT/bin/llvm-config" ]; then
    echo "ERROR: LLVM installation not found at $TOOLROOT"
    echo "Please run ./00-build-llvm-rvv.sh first to build LLVM with RISC-V vector extensions support"
    exit 1
fi

echo "Using LLVM from: $TOOLROOT"
export PATH="$TOOLROOT/bin:$PATH"
$TOOLROOT/bin/llvm-config --version
echo "LLVM supports these targets: $($TOOLROOT/bin/llvm-config --targets-built)"

echo 
echo "=================>>> 2. Python dependencies"
# Create virtual environment for better isolation
VENV_DIR="$SCRIPT_DIR/../data/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install "cython>=0.29" numpy scipy cloudpickle \
    packaging psutil tornado typing_extensions \
    ml_dtypes onnx onnxruntime
python -m pip install --upgrade "onnxscript>=0.1.0" sympy

echo 
echo "=================>>> 3. Downloading TVM"
if [[ -d "$TVM_HOME" ]]; then
    echo "Removing old TVM repo"
    rm -rf "${TVM_HOME:?}"
fi

git clone --recursive https://github.com/apache/tvm.git "$TVM_HOME"
cd "$TVM_HOME"
git checkout "$TVM_VERSION"
git submodule update --init --recursive

# Explicitly set flags to compile for x86-64
export CC="$TOOLROOT/bin/clang -target x86_64-unknown-linux-gnu"
export CXX="$TOOLROOT/bin/clang++ -target x86_64-unknown-linux-gnu"
# Unset any previously set flags that might interfere
unset CFLAGS CXXFLAGS

echo 
echo "=================>>> 4. CMake Configuration"
[[ -d "$Host_BUILD" ]] && rm -rf "${Host_BUILD:?}"
mkdir -p "$Host_BUILD"
cp "${SCRIPT_DIR}"/config.cmake.host "$Host_BUILD/config.cmake"

cd "$Host_BUILD" || exit 1

echo 
echo "=================>>> 5. Building C++ core"
# Ensure we're targeting x86-64 when using our custom LLVM
cmake .. -G Ninja \
      -DCMAKE_C_COMPILER="$TOOLROOT/bin/clang" \
      -DCMAKE_CXX_COMPILER="$TOOLROOT/bin/clang++" \
      -DCMAKE_C_FLAGS="-target x86_64-unknown-linux-gnu" \
      -DCMAKE_CXX_FLAGS="-target x86_64-unknown-linux-gnu" \
      -DLLVM_CONFIG_PATH="$TOOLROOT/bin/llvm-config"

ninja -j "$JOBS"

echo 
echo "=================>>> 6. Building Cython modules and Python wheel"
cd "$TVM_HOME/python"
# Build the Python wheel
python -m pip wheel . -w "$TVM_HOME/dist"
# Install the wheel
python -m pip install --force-reinstall "$TVM_HOME/dist"/tvm-*.whl

echo 
echo "=================>>> 7. Verifying import (Relax should be available)"
export LD_LIBRARY_PATH="$Host_BUILD:$LD_LIBRARY_PATH"
python - <<'PY'
import tvm, ctypes, sys
print("TVM version:", tvm.__version__)
print("Relax in dir:", "relax" in dir(tvm))
print("Loaded lib:", tvm._ffi.base._LIB._name)
print("RUNTIME_ONLY flag:", tvm._ffi.base._RUNTIME_ONLY)
ctypes.CDLL(tvm._ffi.base._LIB._name)   # force-load if not already loaded

# Check for TIR and Relax modules
print("\nVerifying TIR and Relax availability:")
print("TIR available:", hasattr(tvm, "tir"))
print("Relax available:", hasattr(tvm, "relax"))

# Check RISC-V support with the new API
print("\nChecking RISC-V target support:")
try:
    from tvm.target.target import Target
    # For TVM >= 0.20.0, the target string format has changed
    riscv_target = Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=thead-c906 -mattr=+v")
    print("✓ RISC-V target created successfully")
    print(f"Target kind: {riscv_target.kind}")
    print(f"Target attrs: {riscv_target.attrs}")
    print(f"Target keys: {riscv_target.keys}")
except Exception as e:
    print("✗ RISC-V target issue:", str(e))
    
# Try alternative target format if the above fails
try:
    if 'riscv_target' not in locals() or riscv_target is None:
        from tvm.target.target import Target
        riscv_target = Target("llvm", options={"mtriple": "riscv64-unknown-linux-gnu", 
                                             "mcpu": "thead-c906", 
                                             "mattr": "+v"})
        print("✓ RISC-V target created with alternate format")
        print(f"Target kind: {riscv_target.kind}")
        print(f"Target attrs: {riscv_target.attrs}")
except Exception as e:
    print("✗ Alternative RISC-V target definition issue:", str(e))
PY

echo "=================>>> 8. Checking LLVM capabilities for TVM"
python - <<'PY'
import tvm

# Check LLVM version with new API
print("Checking LLVM version:")
try:
    from tvm.target import codegen
    print("LLVM enabled:", codegen.llvm_version_major() > 0)
    print("LLVM version major:", codegen.llvm_version_major())
    # Note: llvm_version_minor might not exist in newer TVM versions
    try:
        print("LLVM version minor:", codegen.llvm_version_minor())
    except AttributeError:
        print("llvm_version_minor not available in this TVM version")
except Exception as e:
    print("Error checking LLVM version:", str(e))

# Check available targets with new API
print("\nChecking available targets:")
try:
    from tvm.target import Target
    print("Available target kinds:")
    for kind in Target.list_kinds():
        print(f"  - {kind}")
    
    # Check if LLVM is available as a target kind
    print("\nLLVM available as target:", "llvm" in Target.list_kinds())
    
    # Try to create an LLVM target for x86 to verify functionality
    x86_target = Target("llvm -mtriple=x86_64-unknown-linux-gnu")
    print("x86_64 LLVM target created successfully")
    
    # Check if we can create a valid RISC-V target
    riscv_target = Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=thead-c906 -mattr=+v")
    print("RISC-V LLVM target created successfully")
    print(f"Target keys: {riscv_target.keys}")
except Exception as e:
    print("Error checking available targets:", str(e))
PY

echo ">>> Done! wheel package is in $TVM_HOME/dist/"

echo
echo "======= Result: ======="
echo "libtvm_allvisible:"
readelf -h "$Host_BUILD/libtvm_allvisible.so"
echo "libtvm_runtime:"
readelf -h "$Host_BUILD/libtvm_runtime.so"
echo "libtvm:"
readelf -h "$Host_BUILD/libtvm.so"
echo "======================="
echo
