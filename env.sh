#!/usr/bin/env bash

# ---------- общие переменные ----------
export ZSTD_LIB_DIR=/usr/lib/x86_64-linux-gnu
export TVM_VERSION="v0.20.0"           # Relay ещё на месте
export TVM_HOME=$HOME/Custler/TVM/tvm-$TVM_VERSION
export Host_BUILD=$TVM_HOME/build
export Dev_BUILD=$TVM_HOME/build-riscv
export CONDA_ENV="tvm20-venv"

# ---------- тулчейн для TH1520 ----------
export TOOLROOT=/opt/riscv
export RISCV_CFLAGS="-march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -O3"   # ← только RISC-V
# ---------- параллелизм ----------
export JOBS=$(nproc)
