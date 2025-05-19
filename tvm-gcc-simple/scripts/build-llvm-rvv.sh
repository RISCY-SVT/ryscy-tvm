#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Параметры сборки
LLVM_URL="https://github.com/dkurt/llvm-rvv-071"
LLVM_VER="rvv-071"
INSTALL_DIR="/opt/riscv"
BUILD_DIR="${SCRIPT_DIR}/../data/build-llvm"
JOBS=$(nproc)

# Установка зависимостей (минимальный набор)
install_dependencies() {
    echo "=== Installing dependencies ==="
    sudo apt-get update
    sudo apt-get install -y build-essential cmake ninja-build git python3 python3-dev \
                         libxml2-dev zlib1g-dev libedit-dev libncurses5-dev
}

# Клонирование LLVM и подмодулей
clone_llvm() {
    echo "=== Cloning LLVM repository ==="
    mkdir -p $(dirname "$BUILD_DIR")
    
    if [ ! -d "$BUILD_DIR/src" ]; then
        git clone --depth 1 --branch ${LLVM_VER} ${LLVM_URL} "$BUILD_DIR/src"
        cd "$BUILD_DIR/src"
        git submodule update --init --recursive --depth 1 --single-branch
    else
        echo "Source directory already exists, skipping clone."
        cd "$BUILD_DIR/src"
        git pull
        git submodule update --init --recursive
    fi
}

# Сборка LLVM
build_llvm() {
    echo "=== Building LLVM with RISC-V support ==="
    mkdir -p "$BUILD_DIR/build"
    cd "$BUILD_DIR/build"
    
    # Исправленная конфигурация - включаем статический анализатор, но убираем другие ненужные компоненты
    cmake -G Ninja "$BUILD_DIR/src/llvm" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf \
        -DLLVM_TARGETS_TO_BUILD="RISCV;X86" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DLLVM_ENABLE_WARNINGS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DCLANG_ENABLE_ARCMT=OFF \
        -DCLANG_ENABLE_STATIC_ANALYZER=ON \
        -DLLVM_ENABLE_Z3_SOLVER=OFF

    ninja -j "$JOBS"
}

# Установка LLVM
install_llvm() {
    echo "=== Installing LLVM to $INSTALL_DIR ==="
    cd "$BUILD_DIR/build"
    sudo mkdir -p "$INSTALL_DIR"
    sudo ninja install
    
    # Создание символических ссылок для удобства
    echo "=== Creating symbolic links ==="
    sudo ln -sf "$INSTALL_DIR/bin/clang" /usr/bin/clang
    sudo ln -sf "$INSTALL_DIR/bin/clang++" /usr/bin/clang++
    sudo ln -sf "$INSTALL_DIR/bin/llvm-config" /usr/bin/llvm-config
    
    echo "=== Installation complete ==="
    echo "LLVM has been installed to $INSTALL_DIR"
    echo "Symlinks created in /usr/bin for clang, clang++, and llvm-config"
    
    # Вывод версии установленного clang и llvm
    "$INSTALL_DIR/bin/clang" --version
    "$INSTALL_DIR/bin/llvm-config" --version
}

# Сборка compiler-rt (библиотеки времени выполнения для RISC-V)
build_compiler_rt() {
    echo "=== Building compiler-rt for RISC-V ==="
    mkdir -p "$BUILD_DIR/compiler-rt-build"
    cd "$BUILD_DIR/compiler-rt-build"
    
    cmake "$BUILD_DIR/src/compiler-rt" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_C_COMPILER_TARGET="riscv64-unknown-elf" \
        -DCMAKE_ASM_COMPILER_TARGET="riscv64-unknown-elf" \
        -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
        -DCOMPILER_RT_BAREMETAL_BUILD=ON \
        -DCOMPILER_RT_BUILD_BUILTINS=ON \
        -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
        -DCOMPILER_RT_BUILD_MEMPROF=OFF \
        -DCOMPILER_RT_BUILD_PROFILE=OFF \
        -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
        -DCOMPILER_RT_BUILD_XRAY=OFF \
        -DCMAKE_C_COMPILER_WORKS=1 \
        -DCMAKE_CXX_COMPILER_WORKS=1 \
        -DCMAKE_C_COMPILER="$INSTALL_DIR/bin/clang" \
        -DCMAKE_C_FLAGS="-march=rv64gc -mabi=lp64d -mno-relax -mcmodel=medany" \
        -DCMAKE_ASM_FLAGS="-march=rv64gc -mabi=lp64d -mno-relax -mcmodel=medany" \
        -DCMAKE_AR="$INSTALL_DIR/bin/llvm-ar" \
        -DCMAKE_NM="$INSTALL_DIR/bin/llvm-nm" \
        -DCMAKE_RANLIB="$INSTALL_DIR/bin/llvm-ranlib" \
        -DLLVM_CMAKE_DIR="$INSTALL_DIR/bin/llvm-config"
    
    ninja -j "$JOBS"
    sudo ninja install
    
    # Создание ссылки для компилятора runtime 
    CLANG_VERSION=$("$INSTALL_DIR/bin/llvm-config" --version | cut -d. -f1)
    sudo mkdir -p "$INSTALL_DIR/lib/clang/$CLANG_VERSION/lib"
    sudo ln -sf "$INSTALL_DIR/lib/linux" "$INSTALL_DIR/lib/clang/$CLANG_VERSION/lib" || true
}

# Очистка временных файлов сборки
cleanup() {
    echo "=== Cleaning up build directory ==="
    sudo rm -rf "$BUILD_DIR/build"
    sudo rm -rf "$BUILD_DIR/compiler-rt-build"
    # Оставляем исходники на случай, если они понадобятся позже
    echo "Build directories removed, source code preserved."
}

# Основная функция
main() {
    # Проверка версии CMake
    CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
    if [ "$(printf '%s\n' "3.20" "$CMAKE_VERSION" | sort -V | head -n1)" != "3.20" ]; then
        echo "Error: CMake version must be at least 3.20, but you have $CMAKE_VERSION"
        exit 1
    fi
    
    install_dependencies
    clone_llvm
    build_llvm
    install_llvm
    build_compiler_rt
    cleanup
    
    echo "=== All done! ==="
    echo "LLVM with RISC-V support (including Vector Extensions) has been successfully built and installed."
}

# Запуск основной функции
main "$@"
