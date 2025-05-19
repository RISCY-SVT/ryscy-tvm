#!/usr/bin/env bash
set -eE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}"/env.sh

INSTALL_DIR="${TOOLROOT}"
BUILD_DIR="${SCRIPT_DIR}/../data/build-llvm"
JOBS=$(nproc)

# Error handling
function handle_error {
    echo "###-ERROR: Error occurred at line $1"
    exit 1
}

# Install dependencies (minimal set)
install_dependencies() {
    echo
    echo "=== Installing dependencies ==="
    sudo apt-get update
    sudo apt-get install -y build-essential cmake ninja-build git python3 python3-dev \
                         libxml2-dev zlib1g-dev libedit-dev libncurses5-dev
}

# Clone LLVM and submodules
clone_llvm() {
    echo
    echo "=== Cloning LLVM repository ==="
    mkdir -p $(dirname "$BUILD_DIR")
    
    if [ ! -d "$BUILD_DIR/src" ]; then
        git clone --depth 1 --branch ${LLVM_VER} ${LLVM_URL} "$BUILD_DIR/src"
        cd "$BUILD_DIR/src"
        git submodule update --init --recursive --depth 1 --single-branch
    else
        echo "Source directory already exists, checking for updates."
        cd "$BUILD_DIR/src"
        git fetch
        
        # Check if local is behind remote
        if [ "$(git rev-parse HEAD)" != "$(git rev-parse @{u})" ]; then
            echo "Updates available. Pulling changes..."
            git pull
            git submodule update --init --recursive
        else
            echo "Source is up to date."
        fi
    fi
    
    # Verify repository
    check_llvm_repo
}

# Verify LLVM repository integrity
check_llvm_repo() {
    cd "$BUILD_DIR/src"
    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo "Error: Failed to verify LLVM repository"
        exit 1
    fi
    echo "LLVM repository verified successfully"
}

# Build LLVM
build_llvm() {
    echo
    echo "=== Building LLVM with RISC-V support ==="
    mkdir -p "$BUILD_DIR/build"
    cd "$BUILD_DIR/build"
    
    # Configuration with static analyzer, but without unnecessary components
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

# Install LLVM
install_llvm() {
    echo
    echo "=== Installing LLVM to $INSTALL_DIR ==="
    cd "$BUILD_DIR/build"
    sudo mkdir -p "$INSTALL_DIR"
    sudo ninja install
    
    # Create symbolic links for convenience
    echo "=== Creating symbolic links ==="
    sudo ln -sf "$INSTALL_DIR/bin/clang" /usr/bin/clang
    sudo ln -sf "$INSTALL_DIR/bin/clang++" /usr/bin/clang++
    sudo ln -sf "$INSTALL_DIR/bin/llvm-config" /usr/bin/llvm-config
    
    echo "=== Installation complete ==="
    echo "LLVM has been installed to $INSTALL_DIR"
    echo "Symlinks created in /usr/bin for clang, clang++, and llvm-config"
    
    # Output version of installed clang and llvm
    echo
    "$INSTALL_DIR/bin/clang" --version
    "$INSTALL_DIR/bin/llvm-config" --version
}

# Build compiler-rt (runtime libraries for RISC-V)
build_compiler_rt() {
    echo
    echo "=== Building compiler-rt for RISC-V with vector extensions ==="
    mkdir -p "$BUILD_DIR/compiler-rt-build"
    rm -rf   "$BUILD_DIR/compiler-rt-build/*"
    cd "$BUILD_DIR/compiler-rt-build"
    
    # Определяем пути к sysroot и библиотекам
    SYSROOT="/opt/riscv/sysroot"
    LIB_PATH="/usr/lib64v_xthead/lp64d"  # Путь относительно sysroot
    
    # Используем стандартные векторные расширения, которые понимает LLVM
    LLVM_MARCH="-march=rv64gcv -mabi=lp64d -O3"
    
    # Строим правильные флаги компилятора с указанием sysroot и пути поиска библиотек
    COMPILE_FLAGS="${LLVM_MARCH} --sysroot=${SYSROOT} -B${SYSROOT}${LIB_PATH}"
    LINK_FLAGS="--sysroot=${SYSROOT} -B${SYSROOT}${LIB_PATH} -L${SYSROOT}${LIB_PATH} -fuse-ld=lld"
    
    # Конфигурируем CMake
    cmake -G Ninja "$BUILD_DIR/src/compiler-rt" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_C_COMPILER="$INSTALL_DIR/bin/clang" \
        -DCMAKE_CXX_COMPILER="$INSTALL_DIR/bin/clang++" \
        -DCMAKE_ASM_COMPILER="$INSTALL_DIR/bin/clang" \
        -DCMAKE_C_COMPILER_TARGET="riscv64-unknown-linux-gnu" \
        -DCMAKE_CXX_COMPILER_TARGET="riscv64-unknown-linux-gnu" \
        -DCMAKE_ASM_COMPILER_TARGET="riscv64-unknown-linux-gnu" \
        -DCMAKE_SYSROOT="${SYSROOT}" \
        -DCMAKE_C_FLAGS="${COMPILE_FLAGS}" \
        -DCMAKE_CXX_FLAGS="${COMPILE_FLAGS}" \
        -DCMAKE_ASM_FLAGS="${COMPILE_FLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${LINK_FLAGS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="${LINK_FLAGS}" \
        -DCMAKE_MODULE_LINKER_FLAGS="${LINK_FLAGS}" \
        -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
        -DCOMPILER_RT_BUILD_BUILTINS=ON \
        -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
        -DCOMPILER_RT_BUILD_XRAY=OFF \
        -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
        -DCOMPILER_RT_BUILD_PROFILE=OFF \
        -DCOMPILER_RT_BUILD_MEMPROF=OFF \
        -DCOMPILER_RT_BUILD_ORC=OFF \
        -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
        -DCMAKE_AR="$INSTALL_DIR/bin/llvm-ar" \
        -DCMAKE_RANLIB="$INSTALL_DIR/bin/llvm-ranlib" \
        -DCMAKE_NM="$INSTALL_DIR/bin/llvm-nm" \
        -DLLVM_CONFIG_PATH="$INSTALL_DIR/bin/llvm-config" \
        -DCMAKE_FIND_ROOT_PATH="${SYSROOT}" \
        -DLLVM_CMAKE_DIR="$INSTALL_DIR/lib/cmake/llvm" \
        -DCOMPILER_RT_OS_DIR="linux" \
        -DCOMPILER_RT_SUPPORTED_ARCH="riscv64"
    
    # Сначала собираем только builtins
    ninja -j "$JOBS" builtins
    sudo ninja install-builtins
    
    # Создаем линки для compiler runtime
    CLANG_VERSION=$("$INSTALL_DIR/bin/llvm-config" --version | cut -d. -f1)
    sudo mkdir -p "$INSTALL_DIR/lib/clang/$CLANG_VERSION/lib/linux"
    # Копируем библиотеки в нужный каталог
    find . -name "*.a" -type f -exec sudo cp {} "$INSTALL_DIR/lib/clang/$CLANG_VERSION/lib/linux/" \;
    
    echo "=== compiler-rt built successfully ==="
}

# Clean up temporary build files
cleanup() {
    echo
    echo "=== Cleaning up build directory ==="
    sudo rm -rf "$BUILD_DIR/build"
    sudo rm -rf "$BUILD_DIR/compiler-rt-build"
    # Preserve source code in case it's needed later
    echo "Build directories removed, source code preserved."
}

# Основная функция
main() {
    # Check CMake version
    CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
    if [ "$(printf '%s\n' "3.20" "$CMAKE_VERSION" | sort -V | head -n1)" != "3.20" ]; then
        echo "###-ERROR: CMake version must be at least 3.20, but you have $CMAKE_VERSION"
        exit 1
    fi
    
    install_dependencies
    clone_llvm
    build_llvm
    install_llvm
    build_compiler_rt
    cleanup
    
    echo
    echo "=== All done! ==="
    echo "LLVM with RISC-V support (including Vector Extensions) has been successfully built and installed."
}

# Запуск основной функции
main "$@"
