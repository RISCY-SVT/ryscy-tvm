#!/usr/bin/env bash
set -Eeo pipefail

########################      ПАРАМЕТРЫ     ########################
source ./env.sh
# Проверяем путь к кросс-компилятору
if [ ! -d "$TOOLROOT" ]; then
    echo "Ошибка: Каталог с toolchain не существует: $TOOLROOT"
    exit 1
fi

# Добавляем toolchain в PATH
export PATH=$TOOLROOT/bin:$PATH

# Проверяем доступность компилятора
if ! command -v riscv64-unknown-linux-gnu-gcc &> /dev/null; then
    echo "Ошибка: Компилятор riscv64-unknown-linux-gnu-gcc не найден в PATH"
    exit 1
fi

echo "=================>>> 1. Подготовка окружения"
# Активируем conda окружение (если нужно)
if [ -n "$CONDA_ENV" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# Создаем модифицированный конфиг для полного рантайма
echo "=================>>> 2. Создание конфигурации для полного рантайма"
[ -d "$Dev_BUILD" ] && rm -rf "${Dev_BUILD:?}"
mkdir -p "$Dev_BUILD"

# Создаем модифицированный конфигурационный файл для полного рантайма
cat > "$Dev_BUILD/config.cmake" << EOF
# Выключаем LLVM для кодогенерации
set(USE_LLVM OFF)

# Включаем Graph Executor и RPC
set(USE_GRAPH_RUNTIME ON)
set(USE_RPC ON)
set(USE_MLIR OFF)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(HIDE_PRIVATE_SYMBOLS OFF)  # Для полного доступа к символам

# Выключаем бэкенды, которые не нужны на RISC-V
set(USE_CUDA OFF)
set(USE_OPENCL OFF)
set(USE_NCCL OFF)
set(USE_MSCCL OFF)
set(USE_NVTX OFF)
set(USE_ROCM OFF)
set(USE_RCCL OFF)
set(USE_OPENCL_ENABLE_HOST_PTR OFF)
set(USE_METAL OFF)
set(USE_VULKAN OFF)

# Выключаем компоненты, которые не нужны на устройстве
set(USE_KHRONOS_SPIRV OFF)
set(USE_SPIRV_KHR_INTEGER_DOT_PRODUCT OFF)
set(USE_CPP_RPC OFF)
set(USE_CPP_RTVM ON)  # Включаем C++ RTVM для полного рантайма
set(USE_IOS_RPC OFF)
set(USE_STACKVM_RUNTIME ON)  # Включаем stackvm для полной функциональности
set(USE_PIPELINE_EXECUTOR ON)  # Включаем pipeline executor
set(USE_PROFILER ON)  # Включаем профилирование для отладки

# Включаем важные компоненты рантайма
set(USE_RANDOM ON)
set(USE_NNPACK OFF)  # Отключаем, т.к. сложно собрать на RISC-V
set(USE_SORT ON)

# Выключаем бэкенды и компоненты, которые не нужны на RISC-V
set(USE_TFLITE OFF)
set(USE_TENSORFLOW_PATH none)
set(USE_FLATBUFFERS_PATH none)
set(USE_EDGETPU OFF)
set(USE_CUDNN OFF)
set(USE_CUDNN_FRONTEND OFF)
set(USE_CUBLAS OFF)
set(USE_MIOPEN OFF)
set(USE_MPS OFF)
set(USE_ROCBLAS OFF)

# Включаем ARM Compute Library только если нужно
set(USE_ARM_COMPUTE_LIB OFF)
set(USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR OFF)

# Отключаем остальные бэкенды, которые не актуальны для RISC-V
set(USE_TENSORRT_CODEGEN OFF)
set(USE_TENSORRT_RUNTIME OFF)
set(USE_MSC OFF)
set(USE_CLML OFF)
set(USE_CLML_GRAPH_EXECUTOR OFF)
set(TVM_DEBUG_WITH_ABI_CHANGE OFF)

# Выключаем VTA, т.к. он не нужен на RISC-V
set(USE_VTA_FSIM OFF)
set(USE_VTA_TSIM OFF)
set(USE_VTA_FPGA OFF)

# Отключаем остальные компоненты, которые не нужны на RISC-V
set(USE_THRUST OFF)
set(USE_CURAND OFF)
set(USE_TF_TVMDSOOP OFF)
set(USE_PT_TVMDSOOP OFF)
set(USE_FALLBACK_STL_MAP OFF)

# Выключаем Hexagon
set(USE_HEXAGON OFF)
set(USE_HEXAGON_SDK /path/to/sdk)
set(USE_HEXAGON_RPC OFF)
set(USE_HEXAGON_ARCH "v68")
set(USE_MRVL OFF)
set(USE_HEXAGON_QHL OFF)
set(USE_BNNS OFF)

# Включаем сборку статической библиотеки для лучшей переносимости
set(BUILD_STATIC_RUNTIME ON)

# Включаем ccache для ускорения сборки
set(USE_CCACHE AUTO)

# Отключаем backtrace для упрощения сборки
set(USE_LIBBACKTRACE OFF)
set(BACKTRACE_ON_SEGFAULT OFF)
set(USE_PAPI OFF)
set(USE_GTEST OFF)

# Выключаем компоненты, которые не нужны на RISC-V
set(USE_CUTLASS OFF)
set(USE_FLASHINFER OFF)
set(SUMMARIZE ON)  # Включаем сводку для отладки

# Дополнительные настройки
set(USE_LIBTORCH OFF)
set(USE_UMA OFF)
set(USE_KALLOC_ALIGNMENT 64)
SET(CMAKE_VS_PLATFORM_NAME_DEFAULT "x64")
SET(CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE "x64")
set(USE_OPENCL_EXTN_QCOM OFF)

# Выключаем backtrace
add_compile_definitions(DMLC_LOG_STACK_TRACE=0)

# Отключаем lld для линковки RISC-V
set(CMAKE_EXE_LINKER_FLAGS "")
set(CMAKE_SHARED_LINKER_FLAGS "")
set(USE_LLD OFF)
set(USE_ALTERNATIVE_LINKER OFF)

# Включаем полную функциональность рантайма (не только runtime_only)
set(USE_RELAY_DEBUG_RUNTIME ON)
EOF

# Переход в директорию сборки
cd "$Dev_BUILD" || exit 1

echo "=================>>> 3. Сборка полного RISC-V runtime"
# Вызываем CMake с нужными флагами для кросс-компиляции
# с поддержкой векторных инструкций
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=$TOOLROOT/bin/riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=$TOOLROOT/bin/riscv64-unknown-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="$RISCV_CFLAGS -fPIC -ftree-vectorize -D__riscv_vector=1" \
    -DCMAKE_CXX_FLAGS="$RISCV_CFLAGS -fPIC -ftree-vectorize -D__riscv_vector=1" \
    -DCMAKE_FIND_ROOT_PATH=$TOOLROOT/riscv64-unknown-linux-gnu \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DUSE_ALTERNATIVE_LINKER=OFF

# Собираем полный рантайм (не только runtime, но и все необходимые компоненты)
echo "Сборка всех компонентов для полного рантайма..."
make -j$JOBS

# Собираем также и статическую библиотеку рантайма
echo "Сборка статической библиотеки..."
make -j$JOBS libtvm_runtime_static

# Создание информации о версии
echo "=================>>> 4. Проверка собранных библиотек"
echo "Динамическая библиотека:"
file "$Dev_BUILD/libtvm_runtime.so"
readelf -A "$Dev_BUILD/libtvm_runtime.so" | grep -E 'Tag_RISCV|ABI|Arch'

echo "Статическая библиотека:"
file "$Dev_BUILD/libtvm_runtime.a"
readelf -A "$Dev_BUILD/libtvm_runtime.a" | grep -E 'Tag_RISCV|ABI|Arch'

echo "=================>>> 5. Создание полного пакета TVM runtime для RISC-V"
# Создаем временную директорию для сборки пакета
WHEEL_BUILD_DIR="$TVM_HOME/python_riscv_full"
rm -rf "$WHEEL_BUILD_DIR" || true
mkdir -p "$WHEEL_BUILD_DIR"

# Копируем полный Python пакет для создания wheel
echo "Копирование Python пакета..."
cp -r "$TVM_HOME/python/tvm" "$WHEEL_BUILD_DIR/"

# Копируем библиотеки
echo "Копирование библиотек..."
mkdir -p "$WHEEL_BUILD_DIR/tvm/lib"
cp "$Dev_BUILD/libtvm_runtime.so" "$WHEEL_BUILD_DIR/tvm/"
cp "$Dev_BUILD/libtvm_runtime.a" "$WHEEL_BUILD_DIR/tvm/lib/"

# Создаем libtvm_pkg_root для определения корня пакета
touch "$WHEEL_BUILD_DIR/tvm/libtvm_pkg_root"

# Модифицируем базовую конфигурацию для работы в режиме рантайма
cat > "$WHEEL_BUILD_DIR/tvm/_ffi/libinfo.py" << 'EOF'
"""Library information."""
import os
import sys
import ctypes

def find_lib_path(name=None, search_path=None, optional=False):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.

    Returns
    -------
    lib_path : str
        Path to the dynamic library.
    """
    # Это полный рантайм для RISC-V
    if not name:
        name = ["libtvm_runtime.so", "libtvm.so"]
    
    # Ищем библиотеку в каталоге пакета tvm
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_search_paths = [os.path.join(curr_path, "..", p) for p in ["", "..", "lib"]]
    
    lib_path = None
    for path in lib_search_paths:
        path = os.path.abspath(path)  # normalized path
        for libname in name:
            full_path = os.path.join(path, libname)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                lib_path = full_path
                break
        if lib_path:
            break
    
    if not lib_path and not optional:
        message = f"Cannot find libraries: {name}\n"
        message += "List of candidates:\n"
        for libname in name:
            for path in lib_search_paths:
                message += os.path.abspath(os.path.join(path, libname)) + "\n"
        raise RuntimeError(message)
    
    return lib_path

def _get_paths():
    """Get the paths for various components in the TVM package"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    root_path = os.path.abspath(os.path.join(curr_path, ".."))
    lib_path = os.path.join(root_path, "libtvm_runtime.so")
    return {
        "root": root_path,
        "runtime": lib_path,
        "lib": lib_path,
    }
EOF

# Создаем setup.py для сборки wheel
cat > "$WHEEL_BUILD_DIR/setup.py" << EOF
import os
import setuptools
from setuptools import find_packages

def get_version():
    return "0.20.0"

setuptools.setup(
    name="tvm",
    version=get_version(),
    license="Apache-2.0",
    description="TVM Full Runtime for RISC-V (LicheePi4A)",
    author="Apache TVM",
    author_email="dev@tvm.apache.org",
    url="https://github.com/apache/tvm",
    packages=find_packages(),
    package_data={
        'tvm': ['*.so', '*.a', 'lib/*.a', 'libtvm_pkg_root'],
    },
    install_requires=[
        "numpy",
        "packaging",
        "scipy",
        "cloudpickle",
        "psutil",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
)
EOF

# Создаем файл setup.cfg для указания платформы
cat > "$WHEEL_BUILD_DIR/setup.cfg" << EOF
[bdist_wheel]
plat-name = manylinux2014_riscv64
EOF

# Переходим в директорию сборки wheel и собираем пакет
cd "$WHEEL_BUILD_DIR"

# Генерируем платформо-специфичное колесо
python setup.py bdist_wheel

# Переносим wheel в общую директорию для дистрибутивов
mkdir -p "$TVM_HOME/dist/riscv64_full"
cp dist/*.whl "$TVM_HOME/dist/riscv64_full/"

echo
echo "=================>>> 6. Создание установочного скрипта для девайса"
cat > "$TVM_HOME/dist/riscv64_full/install_on_device.sh" << 'EOF'
#!/bin/bash
# Скрипт для установки полного TVM runtime на LicheePi4A

# Устанавливаем необходимые зависимости
pip install numpy scipy cloudpickle psutil packaging

# Устанавливаем TVM wheel пакет
pip install --force-reinstall tvm-*.whl

# Проверяем установку
python -c "import tvm; print('TVM версия:', tvm.__version__); \
          print('Пути библиотек:', tvm._ffi.base._LIB._name); \
          print('Полный runtime:', not getattr(tvm._ffi.base, '_RUNTIME_ONLY', True))"

echo "Установка завершена!"
EOF

chmod +x "$TVM_HOME/dist/riscv64_full/install_on_device.sh"

echo
echo "=================>>> Готово!"
echo "Полный TVM RISC-V Runtime wheel находится в $TVM_HOME/dist/riscv64_full/"
echo "Для установки на устройстве LicheePi4A скопируйте содержимое каталога dist/riscv64_full на устройство и выполните:"
echo "bash install_on_device.sh"
echo
echo "Для использования векторных инструкций при инференсе убедитесь, что вы компилируете модели с:"
echo "target = tvm.target.Target(\"llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+v\")"
echo
