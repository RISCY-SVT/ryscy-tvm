#!/usr/bin/env bash
set -eEo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

########################      ПАРАМЕТРЫ     ########################
source ./env.sh
###################################################################

echo "=================>>> 0. Системные пакеты"
sudo apt-get update
sudo apt-get install -y git build-essential cmake ninja-build \
                        llvm-14-dev clang-14 \
                        libedit-dev libxml2-dev zlib1g-dev libz-dev libzstd-dev

echo 
# echo "=================>>> 1. Conda-окружение"
# # source $HOME/anaconda3/bin/activate
# conda create -y -n "$CONDA_ENV" python=3.11
# # shellcheck source=/dev/null
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$CONDA_ENV"
# # <---  добавляем: более свежие компиляторы и RTL  –-->
# conda config --add channels conda-forge
# conda install -y libstdcxx-ng=13  # или просто  conda install -y libstdcxx-ng
# # (опционально, если хотите собирать TVM conda-GCC)
# # conda install -y gcc_linux-64 gxx_linux-64

echo 
echo "=================>>> 2. Python-зависимости"
python -m pip install -U pip setuptools wheel
python -m pip install "cython>=0.29" numpy scipy cloudpickle \
                       packaging psutil tornado typing_extensions \
                       ml_dtypes onnx onnxruntime
python -m pip install --upgrade "onnxscript>=0.1.0" sympy
pip install --upgrade pip

echo 
echo "=================>>> 3. Скачиваем TVM"
if [[ -d "$TVM_HOME" ]];then
    echo "Remome old TVM repo"
    rm -rf "${TVM_HOME:?}"
fi

git clone --recursive https://github.com/apache/tvm.git "$TVM_HOME"
cd "$TVM_HOME"
git checkout "$TVM_VERSION"
git submodule update --init --recursive

# --- гарантируем, что PATH «чистый» ---
CurrPATH="$PATH"
PATH=$(getconf PATH)          # = /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset CFLAGS CXXFLAGS         # (на всякий случай)

echo 
echo "=================>>> 4. Конфигурация CMake"
[[ -d "$Host_BUILD" ]] && rm -rf "${Host_BUILD:?}"
mkdir -p "$Host_BUILD"
cp "${SCRIPT_DIR}"/config.cmake.host "$Host_BUILD/config.cmake"

cd "$Host_BUILD"  || exit 1

echo 
echo "=================>>> 5. Сборка C++ ядра"
# --- только хостовые компиляторы -------------------------------
native_path=$(getconf PATH)          # /usr/local/sbin:/usr/local/bin:...
export PATH="$native_path"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

unset CFLAGS CXXFLAGS
cmake .. -G Ninja \
      -DCMAKE_C_COMPILER=/usr/bin/gcc \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++

ninja -j "$JOBS"          # создаст libtvm.so и libtvm_runtime.so

export PATH=$CurrPATH

echo 
echo "=================>>> 6. Cython-модули + Python-колёсико"
# # restore PATH
# PATH="$CurrPATH"
# export PATH
cd "$TVM_HOME/python"
# conda deactivate
# ВАЖНО: НЕ задаём TVM_LIBRARY_PATH, чтобы собрать ПОЛНЫЙ пакет, а не runtime-only!
# python3 -m 
pip wheel . -w "$TVM_HOME/dist"
# python3 -m 
pip install --force-reinstall "$TVM_HOME/dist"/tvm-*.whl

echo 
echo "=================>>> 7. Проверка импорта (Relax должен быть доступен)"
export LD_LIBRARY_PATH="$Host_BUILD:$LD_LIBRARY_PATH"
python - <<'PY'
import tvm, ctypes, sys
print("TVM version:", tvm.__version__)
print("Relax in dir:", "relax" in dir(tvm))
print("Loaded lib:", tvm._ffi.base._LIB._name)
print("RUNTIME_ONLY flag:", tvm._ffi.base._RUNTIME_ONLY)
ctypes.CDLL(tvm._ffi.base._LIB._name)   # force-load, если не подцепилось
PY

echo ">>> Готово!  whl лежит в  $TVM_HOME/dist/"

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
