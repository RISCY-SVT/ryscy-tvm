#!/usr/bin/env bash

# ---------- общие переменные ----------
export ZSTD_LIB_DIR=/usr/lib/x86_64-linux-gnu
export TVM_VERSION=${TVM_VERSION:-"main"}  
export TVM_HOME=${TVM_HOME:-"$HOME/Custler/TVM/tvm-$TVM_VERSION"}
export Host_BUILD=${Host_BUILD:-"$TVM_HOME/build"}
export Dev_BUILD=${Dev_BUILD:-"$TVM_HOME/build_riscv"}

# ---------- тулчейн для TVM ----------
export LLVM_URL="https://github.com/llvm/llvm-project.git"
export LLVM_VER="release/20.x"
# LLVM_URL="https://github.com/dkurt/llvm-rvv-071"
# LLVM_VER="rvv-071"

# ---------- тулчейн для TH1520 ----------
export TOOLROOT=${TOOLROOT:-"/opt/riscv"}
# export RISCV_CFLAGS=${RISCV_CFLAGS:-"-march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -O3"}   # ← только RISC-V
export RISCV_CFLAGS="-march=rv64gcv_zfh -mabi=lp64d -O3"   # ← только RISC-V

# ---------- параллелизм ----------
export JOBS=${JOBS:-$(nproc)}

# ---------- version info ----------
# Extract TVM version from git if available
# export TVM_GIT_VERSION=${TVM_GIT_VERSION:-$(cd "$TVM_HOME" 2>/dev/null && git describe --tags 2>/dev/null || echo "$TVM_VERSION")}
