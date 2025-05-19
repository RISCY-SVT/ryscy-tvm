#!/usr/bin/env bash

# ---------- общие переменные ----------
export ZSTD_LIB_DIR=/usr/lib/x86_64-linux-gnu
export TVM_VERSION=${TVM_VERSION:-"v0.20.0"}  
export TVM_HOME=${TVM_HOME:-"$HOME/Custler/TVM/tvm-$TVM_VERSION"}
export Host_BUILD=${Host_BUILD:-"$TVM_HOME/build"}
export Dev_BUILD=${Dev_BUILD:-"$TVM_HOME/build-riscv"}

# ---------- тулчейн для TH1520 ----------
export TOOLROOT=${TOOLROOT:-"/opt/riscv"}
export RISCV_CFLAGS=${RISCV_CFLAGS:-"-march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -O3"}   # ← только RISC-V

# ---------- параллелизм ----------
export JOBS=${JOBS:-$(nproc)}
